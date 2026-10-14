// Kimi Delta Attention prompt rows for every CUDA family (float64 reference: core/kda_ref.zig): 16-row WY chunks.

// A block a (segment, head), chunks from its first row: a prompt cut at multiples of 16 rows keeps one call's bits.

// Thread t holds S row t / 2, keys 64 (t & 1) .. + 63; every sum runs in one fixed order.

#include "kda.cuh"

namespace tf_kda {

constexpr int C = 16;     // rows a chunk
constexpr int KH = D / 2; // keys a thread

struct Chunk {
    float q[C][D];  // q, then its decay from the chunk's start times q
    float k[C][D];  // k, then its decay to the chunk's end times k
    float g[C][D];  // decay factors e^g, then the chunk's outputs rounded to bf16
    float w[C][D];  // beta (decay from the start) k, then W = T of that
    float v[C][D];  // v, then beta v, then U = T (beta v)
    float l[C][C];  // beta_i k_i . (decay j+1 .. i) k_j below the diagonal
    float m[C][C];  // q_i . (decay j+1 .. i) k_j on and below it
    float gl[D];    // the chunk's whole decay
    float beta[C];
};

__device__ __forceinline__ void prompt(const Io& io, const Weights& w, const State& st, const Seg* segs, const Params& p) {
    __shared__ __align__(16) Chunk sh;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int h = blockIdx.x, H = p.heads;
    const Seg seg = segs[blockIdx.y];
    const int64_t hd = static_cast<int64_t>(H) * D, c3 = 3 * hd;
    float* S = st.s + (static_cast<int64_t>(seg.slot) * H + h) * D * D;
    const int64_t conv_at = static_cast<int64_t>(seg.slot) * KEEP * c3;
    const int64_t lx = static_cast<int64_t>(seg.slot) * p.log_rows * c3;
    const int64_t lr = static_cast<int64_t>(seg.slot) * p.log_rows * hd;
    const int v = t >> 1, k0 = KH * (t & 1);
    float s[KH];
#pragma unroll
    for (int i = 0; i < KH; i += 4) {
        const float4 x = *reinterpret_cast<const float4*>(S + v * D + k0 + i);
        s[i] = x.x, s[i + 1] = x.y, s[i + 2] = x.z, s[i + 3] = x.w;
    }
    int64_t at[2];
    float win[2][KEEP], cw[2][TAPS];
    channels(w, st, hd, h, conv_at, lx, seg.kept, t, at, win, cw);
    for (int r = 0; r < seg.kept; ++r) { // the accepted rows of the last window, as the rows kernel replays them
        const int64_t a0 = lr + (static_cast<int64_t>(r) * H + h) * D;
        const float u = st.log_u[a0 + v];
#pragma unroll
        for (int i = 0; i < KH; ++i) s[i] = __fmaf_rn(st.log_k[a0 + k0 + i], u, __fmul_rn(s[i], st.log_a[a0 + k0 + i]));
    }
    const float ea = expf(w.a_log[h]);
    const float scale = 0.08838834764831845f; // 1 / sqrt(D)
    const int gd = t % D; // the gate phase's channel: rows 2 e + t / D
    const float dtb = w.dt_bias[h * D + gd];
    float wn[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) wn[e] = w.norm[lane + 32 * e];
    for (int c0 = 0; c0 < seg.rows; c0 += C) {
        const int n = min(C, seg.rows - c0);
        const int64_t row0 = seg.first + c0;
        bf16 xr[2][C]; // every load of the chunk issued before any is used; rows past the segment are zero
        float fr[C * D / NT], zr[2][4], br = 0.f;
#pragma unroll
        for (int c = 0; c < 2; ++c)
#pragma unroll
            for (int i = 0; i < C; ++i) xr[c][i] = i < n ? io.qkv[(row0 + i) * io.qkv_ld + at[c]] : __float2bfloat16_rn(0.f);
#pragma unroll
        for (int e = 0; e < C * D / NT; ++e) {
            const int i = 2 * e + t / D;
            fr[e] = i < n ? __bfloat162float(io.f[(row0 + i) * io.f_ld + h * D + gd]) : 0.f;
        }
        if (t < n) br = __bfloat162float(io.b[(row0 + t) * io.b_ld + h]);
#pragma unroll
        for (int rr = 0; rr < 2; ++rr)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int i = 2 * warp + rr;
                zr[rr][e] = i < n ? __bfloat162float(io.gate[(row0 + i) * io.gate_ld + h * D + lane + 32 * e]) : 0.f;
            }
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            if (c == 1 && t >= D) break;
            const int lc = chan(t, c), part = lc / D, d = lc % D;
            float* dst = part == 0 ? &sh.q[0][d] : part == 1 ? &sh.k[0][d] : &sh.v[0][d];
#pragma unroll
            for (int i = 0; i < C; ++i) {
                float y = 0.f;
                if (i < n) {
                    const float x = __bfloat162float(xr[c][i]);
                    y = conv_silu(win[c], x, cw[c]);
                    shift(win[c], x);
                }
                dst[i * D] = y;
            }
        }
#pragma unroll
        for (int e = 0; e < C * D / NT; ++e) {
            const int i = 2 * e + t / D;
            sh.g[i][gd] = i < n ? expf(log_decay(fr[e], dtb, ea, p.lower_bound)) : 1.f;
        }
        if (t < C) sh.beta[t] = t < n ? sigmoid(br) : 0.f;
        __syncthreads();
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) { // q and k norms, a warp two rows
            const int i = 2 * warp + rr;
            float q[4], k[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) q[e] = sh.q[i][lane + 32 * e], k[e] = sh.k[i][lane + 32 * e];
            l2norm4(q, scale);
            l2norm4(k, 1.f);
#pragma unroll
            for (int e = 0; e < 4; ++e) sh.q[i][lane + 32 * e] = q[e], sh.k[i][lane + 32 * e] = k[e];
        }
        {
            const int i = t / C, j = t % C;
            if (j > i) sh.l[i][j] = 0.f, sh.m[i][j] = 0.f;
            if (j == i) sh.l[i][j] = 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) { // columns j = warp and 15 - warp: e = prod a_(j+1 .. i), a running product per channel
            const int j = jj == 0 ? warp : C - 1 - warp;
            float kj[4], ej[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) kj[e] = sh.k[j][lane + 32 * e], ej[e] = 1.f;
            for (int i = j; i < C; ++i) {
                float a = 0.f, m = 0.f;
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int d = lane + 32 * e;
                    if (i > j) ej[e] = __fmul_rn(ej[e], sh.g[i][d]);
                    const float ke = __fmul_rn(kj[e], ej[e]);
                    a = __fmaf_rn(sh.k[i][d], ke, a);
                    m = __fmaf_rn(sh.q[i][d], ke, m);
                }
                a = warp_sum(a);
                m = warp_sum(m);
                if (lane == 0) {
                    if (j < i) sh.l[i][j] = __fmul_rn(sh.beta[i], a);
                    sh.m[i][j] = m;
                }
            }
        }
        __syncthreads();
        if (t < D) { // a channel's prefix products (q, beta k) and suffix products (the keys' decay to the chunk's end)
            float pre = 1.f;
#pragma unroll
            for (int i = 0; i < C; ++i) {
                pre = __fmul_rn(pre, sh.g[i][t]);
                sh.w[i][t] = __fmul_rn(sh.beta[i], __fmul_rn(pre, sh.k[i][t]));
                sh.q[i][t] = __fmul_rn(pre, sh.q[i][t]);
            }
            sh.gl[t] = pre;
            float suf = 1.f;
#pragma unroll
            for (int i = C - 1; i >= 0; --i) {
                sh.k[i][t] = __fmul_rn(sh.k[i][t], suf);
                suf = __fmul_rn(suf, sh.g[i][t]);
            }
        } else {
#pragma unroll
            for (int i = 0; i < C; ++i) sh.v[i][t - D] = __fmul_rn(sh.beta[i], sh.v[i][t - D]);
        }
        __syncthreads();
        { // (I + L) X = R by forward substitution, a thread a column of [W | U]
            float* col = t < D ? &sh.w[0][t] : &sh.v[0][t - D];
            float x[C];
#pragma unroll
            for (int i = 0; i < C; ++i) {
                float acc = col[i * D];
#pragma unroll
                for (int j = 0; j < i; ++j) acc = __fmaf_rn(-sh.l[i][j], x[j], acc);
                x[i] = acc;
            }
#pragma unroll
            for (int i = 0; i < C; ++i) col[i * D] = x[i];
        }
        __syncthreads();
        float ws[C], qs[C], vn[C];
#pragma unroll
        for (int i = 0; i < C; ++i) ws[i] = 0.f, qs[i] = 0.f;
#pragma unroll
        for (int kk = 0; kk < KH; kk += 4) {
#pragma unroll
            for (int i = 0; i < C; ++i) {
                const float4 a = *reinterpret_cast<const float4*>(&sh.w[i][k0 + kk]);
                const float4 b = *reinterpret_cast<const float4*>(&sh.q[i][k0 + kk]);
                ws[i] = __fmaf_rn(a.x, s[kk], ws[i]), qs[i] = __fmaf_rn(b.x, s[kk], qs[i]);
                ws[i] = __fmaf_rn(a.y, s[kk + 1], ws[i]), qs[i] = __fmaf_rn(b.y, s[kk + 1], qs[i]);
                ws[i] = __fmaf_rn(a.z, s[kk + 2], ws[i]), qs[i] = __fmaf_rn(b.z, s[kk + 2], qs[i]);
                ws[i] = __fmaf_rn(a.w, s[kk + 3], ws[i]), qs[i] = __fmaf_rn(b.w, s[kk + 3], qs[i]);
            }
        }
#pragma unroll
        for (int i = 0; i < C; ++i) { // both halves of a row: low keys + high keys, the same bits in either thread
            const float wo = __shfl_xor_sync(FULL, ws[i], 1), qo = __shfl_xor_sync(FULL, qs[i], 1);
            const float wt = k0 == 0 ? __fadd_rn(ws[i], wo) : __fadd_rn(wo, ws[i]);
            const float qt = k0 == 0 ? __fadd_rn(qs[i], qo) : __fadd_rn(qo, qs[i]);
            vn[i] = __fsub_rn(sh.v[i][v], wt);
            float o = qt;
#pragma unroll
            for (int j = 0; j <= i; ++j) o = __fmaf_rn(sh.m[i][j], vn[j], o);
            if (k0 == 0) sh.g[i][v] = bfr(o); // g is dead: the outputs take its place
        }
#pragma unroll
        for (int kk = 0; kk < KH; kk += 4) {
            const float4 g4 = *reinterpret_cast<const float4*>(&sh.gl[k0 + kk]);
            float x0 = __fmul_rn(s[kk], g4.x), x1 = __fmul_rn(s[kk + 1], g4.y), x2 = __fmul_rn(s[kk + 2], g4.z), x3 = __fmul_rn(s[kk + 3], g4.w);
#pragma unroll
            for (int j = 0; j < C; ++j) {
                const float4 kd = *reinterpret_cast<const float4*>(&sh.k[j][k0 + kk]);
                x0 = __fmaf_rn(kd.x, vn[j], x0), x1 = __fmaf_rn(kd.y, vn[j], x1), x2 = __fmaf_rn(kd.z, vn[j], x2), x3 = __fmaf_rn(kd.w, vn[j], x3);
            }
            s[kk] = x0, s[kk + 1] = x1, s[kk + 2] = x2, s[kk + 3] = x3;
        }
        __syncthreads();
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
            const int i = 2 * warp + rr;
            if (i >= n) break;
            const float rstd = out_rstd(sh.g[i], p.eps, lane);
#pragma unroll
            for (int e = 0; e < 4; ++e)
                io.y[(row0 + i) * io.y_ld + h * D + lane + 32 * e] = gated(sh.g[i][lane + 32 * e], rstd, wn[e], sigmoid(zr[rr][e]));
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < KH; i += 4) *reinterpret_cast<float4*>(S + v * D + k0 + i) = make_float4(s[i], s[i + 1], s[i + 2], s[i + 3]);
#pragma unroll
    for (int c = 0; c < 2; ++c)
        if (c == 0 || t < D) put_window(st, conv_at, c3, at[c], win[c]);
}

}  // namespace tf_kda

using namespace tf_kda;

extern "C" __global__ void __launch_bounds__(NT, 1) tf_kda_prompt(const Io io, const Weights w, const State st, const Seg* segs, const Params p) {
    prompt(io, w, st, segs, p);
}
