// Our own kernels for Kolibri 1's forward: the embedding rows as the fp32 residual (torch's embed[ids].float()).

#include <cuda_bf16.h>
#include <stdint.h>

// out[r, :] = fp32(embed[ids[r], :]): one block a row, exact (bf16 to fp32 widens without rounding).
extern "C" __global__ void tf_kolibri_embed(const __nv_bfloat16* __restrict__ embed, const int* __restrict__ ids, int d,
                                            float* __restrict__ out) {
    const int64_t r = blockIdx.x;
    const __nv_bfloat16* src = embed + (int64_t)ids[r] * d;
    for (int c = threadIdx.x; c < d; c += blockDim.x) out[r * d + c] = __bfloat162float(src[c]);
}
