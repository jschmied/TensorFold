// Products over 128-column blocks, block-FP8 (fp32 scales a 128 x 128 block) or bf16 weights, bf16 rows, fp32 sums.
#include <metal_stdlib>
// Per block: a lane's 4-term FMA chain, the simdgroup sum, times the scale; rows and grouping never change the order.
using namespace metal;
#pragma METAL fp contract(off)

#ifndef TF_CTL
#define TF_CTL 0 // planted faults: 1 next block's scale, 2 codes' sign dropped, 3 gate and up swapped
#endif

struct TfRowsArgs {
  uint rows, k, n, ldx, ldy; // x rows of k (stride ldx), y rows of n (stride ldy)
  uint bf16; // 1: bf16 weights, scales unused
};

// The exact value of an e4m3 code: its seven magnitude bits as a half's exponent and mantissa, times 2^8.
inline float tf_rows_e4m3(uint code) {
  const float v = float(as_type<half>(ushort((code & 0x7Fu) << 7))) * 256.0f;
  if ((code & 0x7Fu) == 0x7Fu) return NAN;
  return (TF_CTL != 2 && (code & 0x80u)) ? -v : v;
}

// Weights k .. k + 3 of row `n`: four e4m3 codes or four bf16 values.
inline float4 tf_rows_w4(const device uchar* W, uint k, uint n, uint c, uint bf16) {
  if (bf16 != 0) {
    const device bfloat* w = (const device bfloat*)W + (size_t)n * k + c;
    return float4(float(w[0]), float(w[1]), float(w[2]), float(w[3]));
  }
  const uint codes = *(const device uint*)(W + (size_t)n * k + c);
  return float4(tf_rows_e4m3(codes & 0xFFu), tf_rows_e4m3((codes >> 8) & 0xFFu),
      tf_rows_e4m3((codes >> 16) & 0xFFu), tf_rows_e4m3(codes >> 24));
}

// One output column `n` of W over `rows` x rows: acc[r] in fp32, block by block.
template <uint R>
inline void tf_rows_column(const device bfloat* X, uint ldx, const device uchar* W, const device float* S, uint k,
    uint n, uint rows, uint bf16, uint lane, thread float* acc) {
  const uint kb = k / 128u;
  for (uint r = 0; r < R; ++r) acc[r] = 0.0f;
  for (uint b = 0; b < kb; ++b) {
    const uint c = b * 128u + lane * 4u;
    const float4 w = tf_rows_w4(W, k, n, c, bf16);
    const uint sb = TF_CTL == 1 ? min(b + 1u, kb - 1u) : b;
    const float s = bf16 != 0 ? 1.0f : S[(n / 128u) * kb + sb];
    for (uint r = 0; r < R; ++r) {
      if (r >= rows) break;
      const device bfloat* x = X + (size_t)r * ldx + c;
      float p = float(x[0]) * w.x;
      p = fma(float(x[1]), w.y, p);
      p = fma(float(x[2]), w.z, p);
      p = fma(float(x[3]), w.w, p);
      acc[r] = bf16 != 0 ? acc[r] + simd_sum(p) : fma(simd_sum(p), s, acc[r]);
    }
  }
}

// Dense rows: y[r][n] = x[r] . W[n], 1-16 rows; four columns a threadgroup of 128, one a simdgroup.
[[kernel, max_total_threads_per_threadgroup(128)]] void tf_block_rows(const device bfloat* X [[buffer(0)]],
    const device uchar* W [[buffer(1)]], const device float* S [[buffer(2)]], constant TfRowsArgs& a [[buffer(3)]],
    device TF_OUT* Y [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint n = tg * 4u + sg;
  if (n >= a.n) return;
  float acc[16];
  tf_rows_column<16>(X, a.ldx, W, S, a.k, n, a.rows, a.bf16, lane, acc);
  if (lane != 0) return;
  for (uint r = 0; r < a.rows; ++r) Y[(size_t)r * a.ldy + n] = TF_OUT(acc[r]);
}

// An expert group: up to 16 pairs of one expert, pairs sorted by expert (moe_sigmoid.metal's tf_moe_plan).
struct TfRowsItem {
  uint expert, first, count;
};

struct TfPickArgs {
  uint k, n, ldx, slots; // x rows of k at stride ldx; a pair p reads x row p / slots
  uint bf16;
};

// Byte offsets into W of an expert's gate, up and down weights and scales: the checkpoint's own tensors, no repack.
struct TfExpertAt {
  ulong gate, gate_s, up, up_s, down, down_s;
};

inline const device float* tf_rows_s(const device uchar* W, ulong at) {
  return (const device float*)(W + at);
}

// Column `n` over up to 16 gathered rows X + xoff[i]: codes and scale decoded once a block, sums as tf_rows_column.
inline void tf_rows_gather(const device bfloat* X, const thread size_t* xoff, uint rows, const device uchar* W,
    const device float* S, uint k, uint n, uint bf16, uint lane, thread float* acc) {
  const uint kb = k / 128u;
  for (uint r = 0; r < rows; ++r) acc[r] = 0.0f;
  for (uint b = 0; b < kb; ++b) {
    const uint c = b * 128u + lane * 4u;
    const float4 w = tf_rows_w4(W, k, n, c, bf16);
    const uint sb = TF_CTL == 1 ? min(b + 1u, kb - 1u) : b;
    const float s = bf16 != 0 ? 1.0f : S[(n / 128u) * kb + sb];
    for (uint r = 0; r < rows; ++r) {
      const device bfloat* x = X + xoff[r] + c;
      float p = float(x[0]) * w.x;
      p = fma(float(x[1]), w.y, p);
      p = fma(float(x[2]), w.z, p);
      p = fma(float(x[3]), w.w, p);
      acc[r] = bf16 != 0 ? acc[r] + simd_sum(p) : fma(simd_sum(p), s, acc[r]);
    }
  }
}

// Gate and up columns `n` of an item's pairs (one expert), bf16(silu(gate) * up) into act[pair][n]; fp32 gate, up.
[[kernel, max_total_threads_per_threadgroup(128)]] void tf_block_glu(const device bfloat* X [[buffer(0)]],
    const device uchar* W [[buffer(1)]], const device TfExpertAt* at [[buffer(2)]],
    const device TfRowsItem* items [[buffer(3)]], const device uint* members [[buffer(4)]],
    const device uint* count [[buffer(5)]], constant TfPickArgs& a [[buffer(6)]], device bfloat* act [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint n = tg.x * 4u + sg;
  if (tg.y >= count[0] || n >= a.n) return;
  const TfRowsItem it = items[tg.y];
  const TfExpertAt e = at[it.expert];
  const uint rows = min(it.count, 16u);
  size_t xoff[16];
  for (uint i = 0; i < rows; ++i) xoff[i] = (size_t)(members[it.first + i] / a.slots) * a.ldx;
  float g[16], u[16];
  tf_rows_gather(X, xoff, rows, W + e.gate, tf_rows_s(W, e.gate_s), a.k, n, a.bf16, lane, g);
  tf_rows_gather(X, xoff, rows, W + e.up, tf_rows_s(W, e.up_s), a.k, n, a.bf16, lane, u);
  if (lane != 0) return;
  for (uint i = 0; i < rows; ++i) {
    const float gi = TF_CTL == 3 ? u[i] : g[i], ui = TF_CTL == 3 ? g[i] : u[i];
    act[(size_t)members[it.first + i] * a.n + n] = bfloat(gi / (1.0f + precise::exp(-gi)) * ui);
  }
}

// Down columns `n` of an item's pairs over their act rows: fp32 y[pair][n].
[[kernel, max_total_threads_per_threadgroup(128)]] void tf_block_down(const device bfloat* act [[buffer(0)]],
    const device uchar* W [[buffer(1)]], const device TfExpertAt* at [[buffer(2)]],
    const device TfRowsItem* items [[buffer(3)]], const device uint* members [[buffer(4)]],
    const device uint* count [[buffer(5)]], constant TfPickArgs& a [[buffer(6)]], device float* Y [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint n = tg.x * 4u + sg;
  if (tg.y >= count[0] || n >= a.n) return;
  const TfRowsItem it = items[tg.y];
  const TfExpertAt e = at[it.expert];
  const uint rows = min(it.count, 16u);
  size_t xoff[16];
  for (uint i = 0; i < rows; ++i) xoff[i] = (size_t)members[it.first + i] * a.k;
  float y[16];
  tf_rows_gather(act, xoff, rows, W + e.down, tf_rows_s(W, e.down_s), a.k, n, a.bf16, lane, y);
  if (lane != 0) return;
  for (uint i = 0; i < rows; ++i) Y[(size_t)members[it.first + i] * a.n + n] = y[i];
}
