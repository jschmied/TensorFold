// Products over 128-column blocks, bf16 rows in, fp32 sums: block-FP8 weights (e4m3 [N][K], fp32 scale_inv
// [N/128][K/128]) or bf16 weights [N][K] (no scales). Every output: per block, a lane's four-term FMA chain, the
// simdgroup's sum, times the block's scale (fp8), added block by block. Row counts and grouping never change it.
#include <metal_stdlib>
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

// Gate and up columns `n` of each pair's expert, bf16(silu(gate) * up) into act[pair][n]; fp32 gate and up.
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
  for (uint i = 0; i < it.count; ++i) {
    const uint p = members[it.first + i];
    const device bfloat* x = X + (size_t)(p / a.slots) * a.ldx;
    float g, u;
    tf_rows_column<1>(x, 0, W + e.gate, tf_rows_s(W, e.gate_s), a.k, n, 1, a.bf16, lane, &g);
    tf_rows_column<1>(x, 0, W + e.up, tf_rows_s(W, e.up_s), a.k, n, 1, a.bf16, lane, &u);
    if (TF_CTL == 3) { const float t = g; g = u; u = t; }
    if (lane == 0) act[(size_t)p * a.n + n] = bfloat(g / (1.0f + precise::exp(-g)) * u);
  }
}

// Down columns `n` of each pair's expert over its act row: fp32 y[pair][n].
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
  for (uint i = 0; i < it.count; ++i) {
    const uint p = members[it.first + i];
    float y;
    tf_rows_column<1>(act + (size_t)p * a.k, 0, W + e.down, tf_rows_s(W, e.down_s), a.k, n, 1, a.bf16, lane, &y);
    if (lane == 0) Y[(size_t)p * a.n + n] = y;
  }
}
