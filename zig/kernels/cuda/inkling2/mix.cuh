// Inkling 2's narrow residual mix, a block a row: ops.cu includes it after its ink2o helpers (bf, tobf, blockSums).
#pragma once

// The block attention-residual mix (Inkling2BlockAttentionResidual) and the norm after it, per row and query.
struct MixArgs {
    const uint16_t* blocks[kMaxBlocks];  // bf16 [rows, C] each: the completed blocks' inputs
    const void* partial;                 // [rows, C]: bf16 when partial_bf16, else fp32
    const float* query;                  // fp32 [nq, C]; null: no mix, the partial alone (layer 0's attention)
    const uint16_t* norm[kMaxQueries];   // bf16 [C] per query: the RMSNorm after the mix
    uint16_t* out;                       // bf16 [nq, rows, C]
    int n_blocks, partial_bf16, nq, rows, C;
    float eps;
};

constexpr int kMixStage = 4, kMixCols = kMixStage * 512;  // tf_ink2_mix: slices of 512 columns staged at a time

__device__ __forceinline__ void mixStage(const MixArgs& a, size_t row, int c0, uint16_t (*sb)[kMixCols], unsigned char* sp) {
    const int cols = min(kMixCols, a.C - c0);  // a multiple of 8: C % 128 == 0 (ops.zig)
    const int chunks = cols / 8;               // 16 bytes of bf16 a chunk
    for (int j = 0; j < a.n_blocks; ++j)
        for (int i = threadIdx.x; i < chunks; i += blockDim.x)
            __pipeline_memcpy_async(&sb[j][i * 8], a.blocks[j] + row * a.C + c0 + i * 8, 16);
    const int pe = a.partial_bf16 ? 2 : 4;
    const unsigned char* p = static_cast<const unsigned char*>(a.partial) + (row * a.C + c0) * pe;
    for (int i = threadIdx.x; i < cols * pe / 16; i += blockDim.x) __pipeline_memcpy_async(sp + i * 16, p + i * 16, 16);
    __pipeline_commit();  // one memory round trip a slice instead of one a block
    __pipeline_wait_prior(0);
    __syncthreads();
}

// Narrow calls (decode rows): staged slices; the same sums in the same order as tf_ink2_mix_rows.
__device__ __forceinline__ void mixRow(const MixArgs& a) {
    constexpr int kV = (kMaxBlocks + 1) * (1 + kMaxQueries);
    __shared__ float sh[32 * kV];
    __shared__ float wts[kMaxQueries][kMaxBlocks + 1];
    __shared__ float rr[kMaxBlocks + 1];
    __shared__ __align__(16) uint16_t sb[kMaxBlocks][kMixCols];
    __shared__ __align__(16) unsigned char sp[kMixCols * 4];
    const size_t row = blockIdx.x;
    const int C = a.C, n = a.n_blocks + 1;
    // block j's column c, staged: thread t reads only the columns t + 512 k it adds, in the order the sums always took
    auto entry = [&](int j, int i) -> float {
        if (j < a.n_blocks) return bf(sb[j][i]);
        return a.partial_bf16 ? bf(reinterpret_cast<const uint16_t*>(sp)[i]) : reinterpret_cast<const float*>(sp)[i];
    };
    if (a.query) {
        // Normalize each source before its query dot; both passes retain the declared strided/shuffle reduction.
        float ss[kMaxBlocks + 1] = {};
        for (int c0 = 0; c0 < C; c0 += kMixCols) {
            mixStage(a, row, c0, sb, sp);
            for (int i = threadIdx.x; i < kMixCols && c0 + i < C; i += blockDim.x) {
#pragma unroll
                for (int j = 0; j < kMaxBlocks + 1; ++j)
                    if (j < n) { const float x = entry(j, i); ss[j] += x * x; }
            }
            __syncthreads();
        }
        blockSums<kMaxBlocks + 1>(ss, sh);
        if (threadIdx.x == 0) for (int j = 0; j < n; ++j) rr[j] = rsqrtf(ss[j] / C + 1e-6f); __syncthreads();
        float v[kV] = {};
        for (int c0 = 0; c0 < C; c0 += kMixCols) {
            mixStage(a, row, c0, sb, sp);
            for (int i = threadIdx.x; i < kMixCols && c0 + i < C; i += blockDim.x) {
                const int c = c0 + i;
                float qc[kMaxQueries];
#pragma unroll
                for (int q = 0; q < kMaxQueries; ++q) qc[q] = q < a.nq ? a.query[static_cast<size_t>(q) * C + c] : 0.0f;
#pragma unroll
                for (int j = 0; j < kMaxBlocks + 1; ++j) if (j < n) {
                    const float normalized = entry(j, i) * rr[j];
#pragma unroll
                    for (int q = 0; q < kMaxQueries; ++q)
                        if (q < a.nq) v[j * (1 + kMaxQueries) + 1 + q] += qc[q] * normalized;
                }
            }
            __syncthreads();
        }
        blockSums<kV>(v, sh);
        if (threadIdx.x == 0) for (int j = 0; j < n; ++j) for (int q = 0; q < a.nq; ++q) wts[q][j] =
            v[j * (1 + kMaxQueries) + 1 + q] * rsqrtf(static_cast<float>(C));
        __syncthreads();
        if (threadIdx.x < a.nq) {
            float* l = wts[threadIdx.x];
            float m = -INFINITY, t = 0.0f;
            for (int j = 0; j < n; ++j) m = fmaxf(m, l[j]);
            for (int j = 0; j < n; ++j) t += expf(l[j] - m);
            for (int j = 0; j < n; ++j) l[j] = expf(l[j] - m) / t;
        }
        __syncthreads();
    }
    // every query's mixed row kept in registers (C <= kMixCache x blockDim, ops.zig): read once, normed, written
    float s[kMaxQueries] = {0.0f, 0.0f, 0.0f};
    float hc[kMaxQueries][kMixCache];
#pragma unroll
    for (int k = 0; k < kMixCache; ++k) {
        const int c = threadIdx.x + k * 512;
        if (k % kMixStage == 0) {
            if (k > 0) __syncthreads();
            if (k * 512 < C) mixStage(a, row, k * 512, sb, sp);
        }
        const int i = threadIdx.x + (k % kMixStage) * 512;
        float h[kMaxQueries] = {0.0f, 0.0f, 0.0f};
        if (c < C) {
            if (a.query) {
#pragma unroll
                for (int j = 0; j < kMaxBlocks + 1; ++j)
                    if (j < n) {
                        const float x = entry(j, i);
#pragma unroll
                        for (int q = 0; q < kMaxQueries; ++q)
                            if (q < a.nq) h[q] += wts[q][j] * (x * rr[j]);
                    }
            } else {
                h[0] = entry(a.n_blocks, i);
            }
        }
#pragma unroll
        for (int q = 0; q < kMaxQueries; ++q) {
            s[q] += h[q] * h[q];  // rows past C add 0.0f * 0.0f: the sums keep their bits
            hc[q][k] = h[q];
        }
    }
    blockSums<kMaxQueries>(s, sh);
#pragma unroll
    for (int q = 0; q < kMaxQueries; ++q) {
        if (q >= a.nq) break;
        const float r = rsqrtf(s[q] / C + a.eps);
        uint16_t* o = a.out + (static_cast<size_t>(q) * a.rows + row) * C;
#pragma unroll
        for (int k = 0; k < kMixCache; ++k) {
            const int c = threadIdx.x + k * 512;
            if (c < C) o[c] = tobf(bf(a.norm[q][c]) * (hc[q][k] * r));
        }
    }
}
