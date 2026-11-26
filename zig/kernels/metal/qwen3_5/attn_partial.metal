// Qwen3.5-2B's decode attention over one key chunk, TF_Q35_ATTN_PARTIAL (first written by gen_qwen35_kernels.py).
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
typedef half float16_t;
#define TF_Q35_ATTN_PARTIAL(NAME) \
[[kernel]] void NAME( \
  const device bfloat16_t* Q [[buffer(0)]], \
  TF_Q35_KV K [[buffer(1)]], \
  TF_Q35_KV V [[buffer(2)]], \
  const constant float* scale [[buffer(3)]], \
  const device int32_t* dims [[buffer(4)]], \
  device float* PM [[buffer(5)]], \
  device float* PL [[buffer(6)]], \
  device float* PO [[buffer(7)]], \
  TF_Q35_EXTRA \
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]], \
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]], \
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) { \
constexpr int D = 256; \
constexpr int G = 4; \
constexpr int CK = 128; \
constexpr int SPLIT = 4; \
constexpr int BLK = 4; \
TF_Q35_CHECK \
 \
  const uint lane = thread_index_in_simdgroup; \
  const uint sgi = simdgroup_index_in_threadgroup; \
  const int g = int(sgi) / SPLIT, s = int(sgi) % SPLIT; \
  const int c = int(threadgroup_position_in_grid.y); \
  const int P = dims[0], W = dims[1], TF_Q35_CAP NCH = dims[3]; \
  const int h = int(threadgroup_position_in_grid.z) / W; \
  const int w = int(threadgroup_position_in_grid.z) % W; \
  constexpr int DPL = D / 32; \
  const int qh = h * G + g; \
  threadgroup float sm[G * SPLIT], sl[G * SPLIT]; \
  threadgroup float so[G * SPLIT][D]; \
  { \
    const int last = P + w; \
    const int k0 = c * CK; \
    const int k1 = min(k0 + CK, last + 1); \
    TF_Q35_CHUNK \
    float q[DPL], o[DPL]; \
    for (int i = 0; i < DPL; i++) { \
      q[i] = float(Q[((qh * dims[5]) + dims[6] + w) * D + int(lane) * DPL + i]); \
      o[i] = 0.0f; \
    } \
    float m = -INFINITY, l = 0.0f; \
    for (int base = k0 + s; base < k1; base += SPLIT * BLK) { \
      float sc[BLK]; \
      int rows[BLK]; \
      float bm = -INFINITY; \
      for (int j = 0; j < BLK; j++) { \
        const int pos = base + j * SPLIT; \
        rows[j] = pos < k1 ? (pos < P ? pos : pos) : -1; \
        float d = 0.0f; \
        if (rows[j] >= 0) { \
          const device bfloat* kr = TF_Q35_ROW(K, kp, rows[j]); \
          for (int i = 0; i < DPL; i++) d = fma(q[i], float(kr[i]), d); \
        } \
        sc[j] = simd_sum(d) * scale[0]; \
        if (rows[j] >= 0) bm = metal::max(bm, sc[j]); \
      } \
      const float mn = metal::max(m, bm); \
      const float a = metal::exp(m - mn); \
      l *= a; \
      for (int i = 0; i < DPL; i++) o[i] *= a; \
      for (int j = 0; j < BLK; j++) { \
        if (rows[j] < 0) continue; \
        const float b = metal::exp(sc[j] - mn); \
        l += b; \
        const device bfloat* vr = TF_Q35_ROW(V, vp, rows[j]); \
        for (int i = 0; i < DPL; i++) o[i] = fma(b, float(vr[i]), o[i]); \
      } \
      m = mn; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lane == 0) { sm[sgi] = m; sl[sgi] = l; } \
    for (int i = 0; i < DPL; i++) so[sgi][int(lane) * DPL + i] = o[i]; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (s == 0) { \
      float mx_ = -INFINITY; \
      for (int t = 0; t < SPLIT; t++) mx_ = metal::max(mx_, sm[g * SPLIT + t]); \
      float lsum = 0.0f, acc[DPL]; \
      for (int i = 0; i < DPL; i++) acc[i] = 0.0f; \
      for (int t = 0; t < SPLIT; t++) { \
        const float e = sl[g * SPLIT + t] > 0.0f ? metal::exp(sm[g * SPLIT + t] - mx_) : 0.0f; \
        lsum = fma(sl[g * SPLIT + t], e, lsum); \
        for (int i = 0; i < DPL; i++) acc[i] = fma(so[g * SPLIT + t][int(lane) * DPL + i], e, acc[i]); \
      } \
      const int slot = (qh * W + w) * NCH + c; \
      if (lane == 0) { PM[slot] = mx_; PL[slot] = lsum; } \
      for (int i = 0; i < DPL; i++) PO[size_t(slot) * D + int(lane) * DPL + i] = acc[i]; \
    } \
  } \
 \
}
// Its knobs for the contiguous cache, K and V [heads, CAP, D]; paged.metal sets KV, EXTRA, CHECK, CAP, CHUNK, ROW.
#ifndef TF_Q35_PAGED
#define TF_Q35_KV const device bfloat16_t*
#define TF_Q35_EXTRA
#define TF_Q35_CHECK
#define TF_Q35_CAP CAP = dims[2],
#define TF_Q35_CHUNK
#define TF_Q35_ROW(KV, at, row) KV + (size_t(h) * CAP + row) * D + int(lane) * DPL
TF_Q35_ATTN_PARTIAL(custom_kernel_qwen35_attn_partial_bfloat16_t_bfloat16_t_bfloat16_t_floatc_int32_t_float_float_float)
#endif
