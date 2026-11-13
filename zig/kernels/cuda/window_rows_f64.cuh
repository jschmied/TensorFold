// window_rows.cuh's float64 replays: a value to bf16 in one rounding, q . k, a head over its keys (moved verbatim).
#pragma once

namespace tf_wrows {

// A float64 value to bf16 in one rounding, ties to even: toward zero to fp32 with a sticky last bit, then nearest.
__device__ __forceinline__ __nv_bfloat16 d2bf(double x) {
    if (x != x) return __float2bfloat16_rn(__int_as_float(0x7fc00000));
    float f = __double2float_rz(x);
    if ((double)f != x && isfinite(f)) f = __int_as_float(__float_as_int(f) | 1);
    return __float2bfloat16_rn(f);
}

// q . k in float64, lanes over the head's dims, then a xor tree; every lane gets the sum.
template <int D>
__device__ __forceinline__ double dot64(const __nv_bfloat16* q, const __nv_bfloat16* k, int lane) {
    double s = 0.0;
#pragma unroll 4
    for (int e = lane; e < D; e += 32) s += (double)__bfloat162float(q[e]) * (double)__bfloat162float(k[e]);
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(FULL, s, o);
    return s;
}

// A head over keys [lo, hi] in float64 for columns c0 + threadIdx.x + NT i < c1 (kv(key, 0): K row, 1: V); a block.
template <int D, typename KV>
__device__ __noinline__ void slow_head(const __nv_bfloat16* qh, KV kv, int lo, int hi, double scale,
                                       __nv_bfloat16* out, int c0, int c1) {
    __shared__ double red[NW];
    __shared__ double pw[NW];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    double m = -INFINITY;
    for (int j = lo + warp; j <= hi; j += NW) m = fmax(m, dot64<D>(qh, kv(j, 0), lane) * scale);
    __syncthreads();
    if (lane == 0) red[warp] = m;
    __syncthreads();
    m = -INFINITY;
    for (int w = 0; w < NW; ++w) m = fmax(m, red[w]);
    constexpr int CPT = D / NT;
    double acc[CPT], l = 0.0;
#pragma unroll
    for (int i = 0; i < CPT; ++i) acc[i] = 0.0;
    for (int jb = lo; jb <= hi; jb += NW) {
        const int j = jb + warp;
        if (j <= hi) {
            const double s = dot64<D>(qh, kv(j, 0), lane) * scale;
            if (lane == 0) pw[warp] = exp(s - m);
        }
        __syncthreads();
        const int nb = min(NW, hi - jb + 1);
        for (int w = 0; w < nb; ++w) {
            const double p = pw[w];
            l += p;
            const __nv_bfloat16* v = kv(jb + w, 1);
#pragma unroll
            for (int i = 0; i < CPT; ++i) {
                const int c = c0 + threadIdx.x + NT * i;
                if (c < c1) acc[i] += p * (double)__bfloat162float(v[c]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < CPT; ++i) {
        const int c = c0 + threadIdx.x + NT * i;
        if (c < c1) out[c] = d2bf(acc[i] / l);
    }
}

} // namespace tf_wrows
