// Kimi Delta Attention decode rows and verify windows for every CUDA family (float64 reference: core/kda_ref.zig).

// A block a (segment, head): the kept logged rows replayed, then each row's conv, gates, delta rule and norm in order.

// Warp w holds S rows 8 w .. 8 w + 7, lane l keys 4 l .. 4 l + 3; sums in one fixed order: a row's bits never vary.

#include "kda.cuh"

namespace tf_kda {

constexpr int RT = 512;        // threads a block
constexpr int RV = D / (RT / 32); // S rows a warp

__device__ __forceinline__ void load4(const float* p, float (&x)[4]) {
    const float4 w = *reinterpret_cast<const float4*>(p);
    x[0] = w.x, x[1] = w.y, x[2] = w.z, x[3] = w.w;
}

__device__ __forceinline__ void store4(float* p, float a, float b, float c, float d) {
    *reinterpret_cast<float4*>(p) = make_float4(a, b, c, d);
}

// One row's inputs as this thread uses them: its raw channel (below 3 D), a forget input, the beta logit, a gate logit.
struct In {
    bf16 x;
    float f, b, z;
};

__device__ __forceinline__ void fetch(const Io& io, int64_t row, int64_t at, int h, int t, int warp, int lane, In& in) {
    in.x = t < 3 * D ? io.qkv[row * io.qkv_ld + at] : __float2bfloat16_rn(0.f);
    in.f = t < D ? __bfloat162float(io.f[row * io.f_ld + h * D + t]) : 0.f;
    in.b = t == 0 ? __bfloat162float(io.b[row * io.b_ld + h]) : 0.f;
    in.z = __bfloat162float(io.gate[row * io.gate_ld + h * D + RV * warp + (lane & (RV - 1))]);
}

__device__ __forceinline__ void rows(const Io& io, const Weights& w, const State& st, const Seg* segs, const Params& p) {
    __shared__ float xs[3][D]; // q, k, v after conv
    __shared__ float as[D];    // decay factors
    __shared__ float os[D];    // the head's output rounded to bf16
    __shared__ float beta_s;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int h = blockIdx.x, H = p.heads;
    const Seg seg = segs[blockIdx.y];
    const int64_t hd = static_cast<int64_t>(H) * D, c3 = 3 * hd;
    float* S = st.s + (static_cast<int64_t>(seg.slot) * H + h) * D * D;
    const int64_t conv_at = static_cast<int64_t>(seg.slot) * KEEP * c3;
    const int64_t lx = static_cast<int64_t>(seg.slot) * p.log_rows * c3;
    const int64_t lr = static_cast<int64_t>(seg.slot) * p.log_rows * hd;
    const bool commit = (seg.flags & SEG_COMMIT) != 0;
    In next;
    const bool chan = t < 3 * D; // a thread a conv channel
    const int tc = chan ? t : 0;
    const int64_t at = (tc / D) * hd + static_cast<int64_t>(h) * D + tc % D;
    float win[KEEP], cw[TAPS];
#pragma unroll
    for (int j = 0; j < TAPS; ++j) cw[j] = w.conv[at * TAPS + j];
    history(st, conv_at, lx, c3, seg.kept, at, win);
    if (seg.rows > 0) fetch(io, seg.first, at, h, t, warp, lane, next);
    float s[RV][4];
#pragma unroll
    for (int j = 0; j < RV; ++j) load4(S + (RV * warp + j) * D + 4 * lane, s[j]);
    const float wn = w.norm[RV * warp + (lane & (RV - 1))];
    for (int r = 0; r < seg.kept; ++r) { // the accepted rows of the last window, as they ran
        const int64_t a0 = lr + (static_cast<int64_t>(r) * H + h) * D;
        float a[4], k[4], u[RV];
        load4(st.log_a + a0 + 4 * lane, a);
        load4(st.log_k + a0 + 4 * lane, k);
#pragma unroll
        for (int j = 0; j < RV; j += 4) {
            const float4 x = *reinterpret_cast<const float4*>(st.log_u + a0 + RV * warp + j);
            u[j] = x.x, u[j + 1] = x.y, u[j + 2] = x.z, u[j + 3] = x.w;
        }
#pragma unroll
        for (int j = 0; j < RV; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) s[j][i] = __fmaf_rn(k[i], u[j], __fmul_rn(s[j][i], a[i]));
    }
    if (seg.kept > 0 && !commit) {
#pragma unroll
        for (int j = 0; j < RV; ++j) store4(S + (RV * warp + j) * D + 4 * lane, s[j][0], s[j][1], s[j][2], s[j][3]);
        if (chan) put_window(st, conv_at, c3, at, win);
    }
    __syncthreads(); // every warp's replay has read the log before row 0 rewrites it
    const float ea = expf(w.a_log[h]);
    const float dtb = t < D ? w.dt_bias[h * D + t] : 0.f;
    const float scale = 0.08838834764831845f; // 1 / sqrt(D)
    for (int r = 0; r < seg.rows; ++r) {
        const In in = next;
        if (r + 1 < seg.rows) fetch(io, seg.first + r + 1, at, h, t, warp, lane, next);
        if (chan) {
            const float x = __bfloat162float(in.x);
            xs[t / D][t % D] = conv_silu(win, x, cw);
            shift(win, x);
            if (!commit) st.log_x[lx + r * c3 + at] = in.x;
        }
        if (t < D) as[t] = expf(log_decay(in.f, dtb, ea, p.lower_bound));
        if (t == 0) beta_s = sigmoid(in.b);
        const float z = sigmoid(in.z);
        __syncthreads();
        float q[4], k[4], a[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) q[i] = xs[0][4 * lane + i], k[i] = xs[1][4 * lane + i], a[i] = as[4 * lane + i];
        l2norm4(q, scale);
        l2norm4(k, 1.f);
        const float beta = beta_s;
        float part[RV], u[RV];
#pragma unroll
        for (int j = 0; j < RV; ++j) {
            part[j] = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s[j][i] = __fmul_rn(s[j][i], a[i]);
                part[j] = __fmaf_rn(s[j][i], k[i], part[j]);
            }
        }
        const float mem = spread<RV>(part, lane);
#pragma unroll
        for (int j = 0; j < RV; ++j) u[j] = __fmul_rn(__fsub_rn(xs[2][RV * warp + j], __shfl_sync(FULL, mem, (32 / RV) * j)), beta);
#pragma unroll
        for (int j = 0; j < RV; ++j) {
            part[j] = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s[j][i] = __fmaf_rn(k[i], u[j], s[j][i]);
                part[j] = __fmaf_rn(s[j][i], q[i], part[j]);
            }
        }
        const float o = spread<RV>(part, lane);
        if ((lane & (32 / RV - 1)) == 0) os[RV * warp + lane / (32 / RV)] = bfr(o);
        if (!commit) {
            const int64_t a0 = lr + (static_cast<int64_t>(r) * H + h) * D;
            if (warp == 0) {
                store4(st.log_a + a0 + 4 * lane, a[0], a[1], a[2], a[3]);
                store4(st.log_k + a0 + 4 * lane, k[0], k[1], k[2], k[3]);
            }
#pragma unroll
            for (int g = 0; g < RV / 4; ++g)
                if (lane == g) store4(st.log_u + a0 + RV * warp + 4 * g, u[4 * g], u[4 * g + 1], u[4 * g + 2], u[4 * g + 3]);
        }
        __syncthreads();
        const float rstd = out_rstd(os, p.eps, lane); // every warp the same sum, then its own RV outputs
        const int v = RV * warp + (lane & (RV - 1));
        if (lane < RV) io.y[(seg.first + r) * io.y_ld + h * D + v] = gated(os[v], rstd, wn, z);
    }
    if (commit) {
#pragma unroll
        for (int j = 0; j < RV; ++j) store4(S + (RV * warp + j) * D + 4 * lane, s[j][0], s[j][1], s[j][2], s[j][3]);
        if (chan) put_window(st, conv_at, c3, at, win);
    }
}

}  // namespace tf_kda

using namespace tf_kda;

extern "C" __global__ void __launch_bounds__(RT) tf_kda_rows(const Io io, const Weights w, const State st, const Seg* segs, const Params p) {
    rows(io, w, st, segs, p);
}
