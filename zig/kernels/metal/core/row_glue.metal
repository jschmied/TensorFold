// Row glue at runtime widths: embed widen, RMS norms (plain, sandwich add), q/k norms + RoPE into a cache; fp32 sums.
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp contract(off)

#ifndef TF_CTL
#define TF_CTL 0 // planted faults: 1 RoPE halves swapped, 2 norm without eps, 3 cache slot off by one
#endif

struct TfGlueArgs {
  uint rows, width; // rows of `width`
  float eps;
};

// A threadgroup's sum of one value a thread (256 threads, eight simdgroups), the same order every call.
inline float tf_glue_sum(float v, threadgroup float* part, uint tid) {
  v = simd_sum(v);
  if ((tid & 31u) == 0) part[tid >> 5] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = 0.0f;
  for (uint i = 0; i < 8u; ++i) t += part[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return t;
}

inline float tf_glue_inv(float ss, uint width, float eps) {
  return precise::rsqrt(ss / float(width) + (TF_CTL == 2 ? 0.0f : eps));
}

// res[r] = table[ids[r]] widened exactly; one thread an element.
[[kernel]] void tf_glue_embed(const device uint* ids [[buffer(0)]], const device bfloat* table [[buffer(1)]],
    constant TfGlueArgs& a [[buffer(2)]], device float* res [[buffer(3)]], uint2 id [[thread_position_in_grid]]) {
  if (id.x >= a.width || id.y >= a.rows) return;
  res[(size_t)id.y * a.width + id.x] = float(table[(size_t)ids[id.y] * a.width + id.x]);
}

// out[r] = bf16(x[r] * rsqrt(mean(x^2) + eps) * w), x fp32; one threadgroup of 256 a row.
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_glue_rms(const device float* x [[buffer(0)]],
    const device bfloat* w [[buffer(1)]], constant TfGlueArgs& a [[buffer(2)]], device bfloat* out [[buffer(3)]],
    uint r [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float part[8];
  const device float* xr = x + (size_t)r * a.width;
  float ss = 0.0f;
  for (uint i = tid; i < a.width; i += 256u) ss = fma(xr[i], xr[i], ss);
  const float inv = tf_glue_inv(tf_glue_sum(ss, part, tid), a.width, a.eps);
  for (uint i = tid; i < a.width; i += 256u) out[(size_t)r * a.width + i] = bfloat(xr[i] * inv * float(w[i]));
}

// res[r] += rms(x[r]) * w1 (x bf16, res fp32 in place), then out[r] = bf16(rms(res[r]) * w2).
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_glue_add_rms(const device bfloat* x [[buffer(0)]],
    device float* res [[buffer(1)]], const device bfloat* w1 [[buffer(2)]], const device bfloat* w2 [[buffer(3)]],
    constant TfGlueArgs& a [[buffer(4)]], device bfloat* out [[buffer(5)]], uint r [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float part[8];
  const device bfloat* xr = x + (size_t)r * a.width;
  device float* rr = res + (size_t)r * a.width;
  float ss = 0.0f;
  for (uint i = tid; i < a.width; i += 256u) ss = fma(float(xr[i]), float(xr[i]), ss);
  const float inv = tf_glue_inv(tf_glue_sum(ss, part, tid), a.width, a.eps);
  float s2 = 0.0f;
  for (uint i = tid; i < a.width; i += 256u) {
    const float v = rr[i] + float(xr[i]) * inv * float(w1[i]);
    rr[i] = v;
    s2 = fma(v, v, s2);
  }
  const float inv2 = tf_glue_inv(tf_glue_sum(s2, part, tid), a.width, a.eps);
  for (uint i = tid; i < a.width; i += 256u) out[(size_t)r * a.width + i] = bfloat(rr[i] * inv2 * float(w2[i]));
}

struct TfQkArgs {
  uint rows, heads, kv_heads, dim; // qkv rows [rows][(heads + 2 kv_heads) dim]
  uint rope, ring; // rope 0: no positions; ring 0: cache row = pos, else pos % ring
  float eps;
};

// One simdgroup a (row, head): q/k normed, neox RoPE by fp32 [pos][dim] tables, rounded once; k, v into cache rows.
[[kernel]] void tf_glue_qk(const device bfloat* qkv [[buffer(0)]], const device bfloat* qn [[buffer(1)]],
    const device bfloat* kn [[buffer(2)]], const device float* cos_t [[buffer(3)]],
    const device float* sin_t [[buffer(4)]], const device uint* pos [[buffer(5)]], constant TfQkArgs& a [[buffer(6)]],
    device bfloat* q [[buffer(7)]], device bfloat* kc [[buffer(8)]], device bfloat* vc [[buffer(9)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint r = tg.y, h = tg.x, d = a.dim, per = d / 32u;
  if (r >= a.rows || h >= a.heads + 2u * a.kv_heads) return;
  const device bfloat* src = qkv + ((size_t)r * (a.heads + 2u * a.kv_heads) + h) * d;
  const uint p = pos[r], slot = (a.ring == 0 ? p : p % a.ring) + (TF_CTL == 3 ? 1u : 0u);
  if (h >= a.heads + a.kv_heads) { // v: as is
    device bfloat* dst = vc + ((size_t)slot * a.kv_heads + h - a.heads - a.kv_heads) * d;
    for (uint i = 0; i < per; ++i) dst[lane + 32u * i] = src[lane + 32u * i];
    return;
  }
  const bool is_q = h < a.heads;
  const device bfloat* w = is_q ? qn : kn;
  float y[8]; // dim <= 256
  float ss = 0.0f;
  for (uint i = 0; i < per; ++i) { y[i] = float(src[lane + 32u * i]); ss = fma(y[i], y[i], ss); }
  const float inv = tf_glue_inv(simd_sum(ss), d, a.eps);
  for (uint i = 0; i < per; ++i) y[i] = y[i] * inv * float(w[lane + 32u * i]);
  device bfloat* dst = is_q ? q + ((size_t)r * a.heads + h) * d : kc + ((size_t)slot * a.kv_heads + h - a.heads) * d;
  if (a.rope == 0) {
    for (uint i = 0; i < per; ++i) dst[lane + 32u * i] = bfloat(y[i]);
    return;
  }
  const uint half_i = per / 2u; // columns i and i + dim/2 sit at y[j] and y[j + half_i]
  for (uint j = 0; j < per; ++j) {
    const uint col = lane + 32u * j;
    const bool lo = j < half_i;
    const float other = lo ? -y[j + half_i] : y[j - half_i];
    const float c = cos_t[(size_t)p * d + col], s = sin_t[(size_t)p * d + col];
    const float rot = TF_CTL == 1 ? -other : other;
    dst[col] = bfloat(y[j] * c + rot * s);
  }
}
