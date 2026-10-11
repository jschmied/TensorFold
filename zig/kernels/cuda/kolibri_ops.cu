// Our own kernels for Kolibri 1: the embedding rows as the fp32 residual, the head's argmax and its bf16 logits.

#include <cuda_bf16.h>
#include <stdint.h>

// out[r, :] = fp32(embed[ids[r], :]): one block a row, exact (bf16 to fp32 widens without rounding).
extern "C" __global__ void tf_kolibri_embed(const __nv_bfloat16* __restrict__ embed, const int* __restrict__ ids, int d,
                                            float* __restrict__ out) {
    const int64_t r = blockIdx.x;
    const __nv_bfloat16* src = embed + (int64_t)ids[r] * d;
    for (int c = threadIdx.x; c < d; c += blockDim.x) out[r * d + c] = __bfloat162float(src[c]);
}

// out[row] = the lowest column holding row's largest fp32 logit (torch.argmax's choice): one block a row.
extern "C" __global__ void __launch_bounds__(1024) tf_kolibri_argmax(const float* __restrict__ logits, int v,
                                                                      unsigned* __restrict__ out) {
    const float* x = logits + (int64_t)blockIdx.x * v;
    float bv = -INFINITY;
    int bi = v;
    for (int i = threadIdx.x; i < v; i += blockDim.x) {
        const float a = x[i];
        if (a > bv || (a == bv && i < bi)) { bv = a; bi = i; }
    }
    for (int d = 16; d > 0; d >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, bv, d);
        const int oi = __shfl_down_sync(0xffffffffu, bi, d);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    __shared__ float sv[32];
    __shared__ int si[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { sv[warp] = bv; si[warp] = bi; }
    __syncthreads();
    if (warp == 0) {
        bv = lane < (int)(blockDim.x >> 5) ? sv[lane] : -INFINITY;
        bi = lane < (int)(blockDim.x >> 5) ? si[lane] : v;
        for (int d = 16; d > 0; d >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, bv, d);
            const int oi = __shfl_down_sync(0xffffffffu, bi, d);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) out[blockIdx.x] = (unsigned)bi;
    }
}

// y = bf16(x), round to nearest even: the logits sample.cu's keyed draws read.
extern "C" __global__ void tf_kolibri_to_bf16(const float* __restrict__ x, __nv_bfloat16* __restrict__ y, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x)
        y[i] = __float2bfloat16_rn(x[i]);
}
