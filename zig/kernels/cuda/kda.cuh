// Kimi Delta Attention's shared device code: the buffers, gate math, conv window, sums and the gated output norm.

#pragma once

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace tf_kda {

constexpr int D = 128;     // key and value dims a head
constexpr int TAPS = 4;    // conv taps
constexpr int KEEP = 3;    // raw rows the conv window keeps
constexpr int NT = 256;    // threads a block
constexpr int SEG_COMMIT = 1;
constexpr unsigned FULL = 0xffffffffu;

typedef __nv_bfloat16 bf16;

// A call's rows (strides in elements): raw q | k | v (H D each), forget inputs, beta and gate logits, the output.
struct Io {
    const bf16* qkv;
    long long qkv_ld;
    const bf16* f;
    long long f_ld;
    const bf16* b;
    long long b_ld;
    const bf16* gate;
    long long gate_ld;
    bf16* y;
    long long y_ld;
};

// conv [3 H D][TAPS] fp32 (q, k, v channels), A_log [H], dt_bias [H D], the output norm's weight [D].
struct Weights {
    const float* conv;
    const float* a_log;
    const float* dt_bias;
    const float* norm;
};

// Slot-major arenas: S [H][D v][D k] fp32, conv [KEEP][3 H D] bf16 oldest first, and the logged window's rows.
struct State {
    float* s;
    bf16* conv;
    float* log_a; // [log_rows][H][D] decay factors
    float* log_k; // [log_rows][H][D] normalized keys
    float* log_u; // [log_rows][H][D] beta (v - S^T k)
    bf16* log_x;  // [log_rows][3 H D] raw rows
};

// One stream's rows in a call: replay `kept` logged rows first, then rows [first, first + rows).
struct Seg {
    int first, rows, kept, flags, slot;
};

// lower_bound < 0: g = lower_bound sigmoid(e^A (f + dt_bias)); else g = -e^A softplus(f + dt_bias).
struct Params {
    int heads, log_rows;
    float lower_bound, eps;
};

// Exactly rounded division on purpose: the same bits on every GPU, where div.full and ex2.approx are not.
__device__ __forceinline__ float sigmoid(float x) { return __fdiv_rn(1.f, __fadd_rn(1.f, expf(-x))); }

__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// A channel's conv over its window (oldest first) and the new raw value, then SiLU rounded to bf16.
__device__ __forceinline__ float conv_silu(const float (&win)[KEEP], float x, const float* w) {
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < KEEP; ++j) acc = __fmaf_rn(win[j], w[j], acc);
    acc = __fmaf_rn(x, w[KEEP], acc);
    return bfr(__fmul_rn(acc, sigmoid(acc)));
}

// The log decay of one channel from its forget input.
__device__ __forceinline__ float log_decay(float f, float dt_bias, float ea, float lb) {
    const float x = __fadd_rn(f, dt_bias);
    if (lb < 0.f) return __fmul_rn(lb, sigmoid(__fmul_rn(ea, x)));
    const float sp = x > 20.f ? x : log1pf(expf(x));
    return -__fmul_rn(ea, sp);
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int m = 16; m; m >>= 1) x = __fadd_rn(x, __shfl_xor_sync(FULL, x, m));
    return x;
}

// N per-lane partials summed across the warp in one fixed order; lanes 32 j / N .. 32 (j + 1) / N - 1 hold total j.
template <int N>
__device__ __forceinline__ float spread(float (&v)[N], int lane) {
#pragma unroll
    for (int m = 16, h = N / 2; m > 0; m >>= 1, h >>= 1) {
        if (h >= 1) {
            const bool up = lane & m;
#pragma unroll
            for (int i = 0; i < h; ++i) {
                const float give = up ? v[i] : v[h + i], keep = up ? v[h + i] : v[i];
                v[i] = __fadd_rn(keep, __shfl_xor_sync(FULL, give, m));
            }
        } else {
            v[0] = __fadd_rn(v[0], __shfl_xor_sync(FULL, v[0], m));
        }
    }
    return v[0];
}

// A head's q (scaled by 1 / sqrt(D)) or k: x / sqrt(sum x^2 + 1e-6), a lane's 4 channels, sums in one order.
__device__ __forceinline__ void l2norm4(float (&x)[4], float scale) {
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) ss = __fmaf_rn(x[i], x[i], ss);
    const float r = __fsqrt_rn(__fadd_rn(warp_sum(ss), 1e-6f));
#pragma unroll
    for (int i = 0; i < 4; ++i) x[i] = scale == 1.f ? __fdiv_rn(x[i], r) : __fmul_rn(__fdiv_rn(x[i], r), scale);
}

// 1 / sqrt(mean(o^2) + eps) of a head's output row in shared memory, a lane's channels lane + 32 i, every warp alike.
__device__ __forceinline__ float out_rstd(const float* o, float eps, int lane) {
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) ss = __fmaf_rn(o[lane + 32 * i], o[lane + 32 * i], ss);
    return __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(__fdiv_rn(warp_sum(ss), static_cast<float>(D)), eps)));
}

// The gated norm's output: o / rms * w * sigmoid(gate), in that order.
__device__ __forceinline__ bf16 gated(float o, float rstd, float w, float z) { return __float2bfloat16_rn(__fmul_rn(__fmul_rn(__fmul_rn(o, rstd), w), z)); }

// A channel's window before a segment's rows: the last KEEP of the conv state then the replayed logged rows.
__device__ __forceinline__ void history(const State& st, int64_t conv_at, int64_t log_at, int64_t c3, int kept, int ch,
                                        float (&win)[KEEP]) {
#pragma unroll
    for (int j = 0; j < KEEP; ++j) {
        const int h = kept + j; // position in conv state ++ logged rows
        win[j] = __bfloat162float(h < KEEP ? st.conv[conv_at + h * c3 + ch] : st.log_x[log_at + (h - KEEP) * c3 + ch]);
    }
}

__device__ __forceinline__ void put_window(const State& st, int64_t conv_at, int64_t c3, int ch, const float (&win)[KEEP]) {
#pragma unroll
    for (int j = 0; j < KEEP; ++j) st.conv[conv_at + j * c3 + ch] = __float2bfloat16_rn(win[j]);
}

__device__ __forceinline__ void shift(float (&win)[KEEP], float x) {
#pragma unroll
    for (int j = 0; j + 1 < KEEP; ++j) win[j] = win[j + 1];
    win[KEEP - 1] = x;
}

// A thread's conv channel c: t, then t + NT below D (else t again, its stores skipped by the caller).
__device__ __forceinline__ int chan(int t, int c) { return c == 1 && t < D ? t + NT : t; }

// A thread's two conv channels: their columns, taps and windows after the replayed rows.
__device__ __forceinline__ void channels(const Weights& w, const State& st, int64_t hd, int h, int64_t conv_at, int64_t lx,
                                         int kept, int t, int64_t (&at)[2], float (&win)[2][KEEP], float (&cw)[2][TAPS]) {
#pragma unroll
    for (int c = 0; c < 2; ++c) {
        const int lc = chan(t, c);
        at[c] = (lc / D) * hd + static_cast<int64_t>(h) * D + lc % D;
#pragma unroll
        for (int j = 0; j < TAPS; ++j) cw[c][j] = w.conv[at[c] * TAPS + j];
        history(st, conv_at, lx, 3 * hd, kept, at[c], win[c]);
    }
}

}  // namespace tf_kda
