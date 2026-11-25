// sparse.py's DSA indexer glue by hand: the index key write, the 4-token pools and the per-row top-512 pools.
#pragma once

#include "glm_pdl.cuh"

#include "glm_triton.cuh"
#include "glm_index_rope.cuh"
#include "common/topk_rows.cuh"

using namespace glm_triton;

// sparse._index_write, a warp a row (D 128): LayerNorm of k_raw, dims below `rope` roped, into ik; gate row into ig.
extern "C" __global__ void __launch_bounds__(32)
    tf_glm_index_write(const __nv_bfloat16* kr, int64_t k_stride, const float* gr, const __nv_bfloat16* lnw,
                       const __nv_bfloat16* lnb, __nv_bfloat16* ik, __nv_bfloat16* ig, const int* pos, float eps,
                       int rope, const float* inv) {
  glm_pdl::enter();
  constexpr int D = 128;
  const int64_t r = blockIdx.x, at = (int64_t)pos[0] + r;
  const int d0 = threadIdx.x * 4;
  float x[4];
  for (int j = 0; j < 4; ++j) x[j] = bf(kr[r * k_stride + d0 + j]);
  float s = x[0] + x[1];
  s = x[2] + s;
  s = x[3] + s;
  for (int o = 16; o > 0; o >>= 1) s = s + __shfl_xor_sync(0xffffffffu, s, o);
  const float mean = div_full(s, (float)D);
  float xc[4];
  for (int j = 0; j < 4; ++j) xc[j] = __fsub_rn(x[j], mean);  // sub.rn in Triton: never contracted
  float v = fmul(xc[1], xc[1]);
  v = __fmaf_rn(xc[0], xc[0], v);
  v = __fmaf_rn(xc[2], xc[2], v);
  v = __fmaf_rn(xc[3], xc[3], v);
  for (int o = 16; o > 0; o >>= 1) v = v + __shfl_xor_sync(0xffffffffu, v, o);
  const float sd = sqrt_approx(fadd(eps, div_full(v, (float)D)));
  float y[4];
  for (int j = 0; j < 4; ++j) y[j] = __fmaf_rn(div_full(xc[j], sd), bf(lnw[d0 + j]), bf(lnb[d0 + j]));
  if (d0 < rope)  // rope % 4 == 0: both of this lane's pairs
    for (int j = 0; j < 4; j += 2) glm_rope::pair(y[j], y[j + 1], at, inv, (d0 + j) / 2);
  for (int j = 0; j < 4; ++j) {
    const int d = d0 + j;
    ik[at * D + d] = __float2bfloat16_rn(y[j]);
    ig[at * D + d] = __float2bfloat16_rn(gr[r * D + d]);
  }
}

// sparse._pool_keys, a warp a pool: softmax of its 4 gate rows (+ ape) times keys, in order; `pool` 1: keys as written.
extern "C" __global__ void __launch_bounds__(32)
    tf_glm_pool_keys(const __nv_bfloat16* ik, const __nv_bfloat16* ig, const __nv_bfloat16* ape, __nv_bfloat16* pk,
                     const int* pos, int64_t rows, const int* pt, int64_t ps, int pool) {
  glm_pdl::enter();
  constexpr int D = 128;
  const int64_t P = pos[0], p = P / pool + blockIdx.x;
  if (pool * p + pool - 1 > P + rows - 1) return;
  const int64_t pp = 512 / pool;  // a page holds 512 rows' pools
  __nv_bfloat16* out = pk + (pt ? (int64_t)pt[p / pp] * (ps / 2) + (p % pp) * D : p * D);
  if (pool == 1) {
    for (int j = 0; j < 4; ++j) out[threadIdx.x * 4 + j] = ik[p * D + threadIdx.x * 4 + j];
    return;
  }
  for (int j = 0; j < 4; ++j) {
    const int d = threadIdx.x * 4 + j;
    float l[4], e[4];
    for (int q = 0; q < 4; ++q) l[q] = bf(ig[(4 * p + q) * D + d]) + bf(ape[q * D + d]);
    const float m = fmaxf(fmaxf(l[0], l[1]), fmaxf(l[2], l[3]));
    for (int q = 0; q < 4; ++q) e[q] = exp_approx(l[q] - m);
    const float s = ((e[0] + e[1]) + e[2]) + e[3];
    float t = 0.0f;
    for (int q = 0; q < 4; ++q) {
      const float tq = rbf(__fmul_rn(rbf(div_full(e[q], s)), bf(ik[(4 * p + q) * D + d])));
      t = q == 0 ? tq : t + tq;
    }
    out[d] = __float2bfloat16_rn(t);
  }
}

namespace glm_index {

// sparse._select_rows (VISIBLE), a row per T-thread block: K best visible pools (tf_topk::select_row), ties to lower.
template <int T>
__device__ __forceinline__ void select_row(const float* scores, int64_t* out, int64_t np, const int* pos, int k,
                                           int pool) {
  const int64_t r = blockIdx.x;
  const int64_t vis = ((int64_t)pos[0] + r + 1) / pool;
  const int64_t limit = np < (vis > k ? vis : k) ? np : (vis > k ? vis : k);
  tf_topk::select_row<T>(scores + r * np, limit, k, out + r * k);
}

}  // namespace glm_index

// sparse._select_rows (VISIBLE), a row per 4-warp block: K best visible pools by 8-bit radix select, ties to lower.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_select_rows(const float* scores, int64_t* out, int64_t np, const int* pos, int k, int pool) {
  glm_pdl::enter();
  glm_index::select_row<128>(scores, out, np, pos, k, pool);
}

// tf_glm_select_rows a row per 1,024-thread block, for a few rows over many pools (a decode step's): its bytes.
extern "C" __global__ void __launch_bounds__(1024)
    tf_glm_select_rows_wide(const float* scores, int64_t* out, int64_t np, const int* pos, int k, int pool) {
  glm_pdl::enter();
  glm_index::select_row<1024>(scores, out, np, pos, k, pool);
}
