// LFM3-VL's gated 3x3 depthwise conv over merge-order rows: Y = bf16(C * bf16(conv3x3(bf16(B * x)))), zero edges.
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

#include "glue_math.cuh"

namespace tf_vit_dwconv {

// The merge-order row at grid (y, x): merge x merge blocks, row by row, each block's patches row by row.
__device__ __forceinline__ int row_at(int y, int x, int bw, int merge) {
    return ((y / merge) * bw + x / merge) * merge * merge + (y % merge) * merge + x % merge;
}

// A thread a (row, channel), taps in (ky, kx) order, fp32 products added in order; CTL 1 reads raster neighbours.
template <int CTL>
__device__ __forceinline__ void dwconv3(const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ W,
                                        __nv_bfloat16* __restrict__ Y, int gh, int gw, int C, int merge) {
    const int n = blockIdx.x, c = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int bw = gw / merge, mm = merge * merge, b = n / mm, r = n % mm;
    const int y = CTL == 1 ? n / gw : (b / bw) * merge + r / merge;
    const int x = CTL == 1 ? n % gw : (b % bw) * merge + r % merge;
    const int64_t C3 = 3LL * C;
    float acc = 0.0f;
#pragma unroll
    for (int ky = 0; ky < 3; ++ky) {
#pragma unroll
        for (int kx = 0; kx < 3; ++kx) {
            const int yy = y + ky - 1, xx = x + kx - 1;
            if (yy < 0 || yy >= gh || xx < 0 || xx >= gw) continue;
            const int m = CTL == 1 ? yy * gw + xx : row_at(yy, xx, bw, merge);
            const float bx = tf_glue::rbf(__fmul_rn(tf_glue::bf(P[m * C3 + c]), tf_glue::bf(P[m * C3 + 2 * C + c])));
            acc = __fadd_rn(acc, __fmul_rn(tf_glue::bf(W[c * 9 + ky * 3 + kx]), bx));
        }
    }
    Y[(int64_t)n * C + c] = __float2bfloat16_rn(__fmul_rn(tf_glue::bf(P[n * C3 + C + c]), tf_glue::rbf(acc)));
}

}  // namespace tf_vit_dwconv
