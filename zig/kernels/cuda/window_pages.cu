// window_rows.cuh over paged caches (Dory's and Spark2.5's full attention at 256, Gemma 4's global at 512), the writer.

#include "window_rows.cuh"
#include "window_rows512.cuh"

using namespace tf_wrows;

// merge's replay of head hd of row r, key head kh over the row's pages in float64 (a block; every thread calls it).
template <int D>
__device__ __noinline__ void tf_wrows::merge_pg_odd(const __nv_bfloat16* q, const __nv_bfloat16* kc,
                                                    const __nv_bfloat16* vc, const int* pos, const int* slot, int h,
                                                    int hk, int win, float scale, Pg pg, __nv_bfloat16* oh, int64_t r,
                                                    int kh, int hd) {
    const int p = pos[r], G = h / hk, ts = (slot != nullptr ? slot[r] : 0) * pg.stride;
    const auto kv = [&](int key, int which) {
        const int64_t pid = pg.tab[ts + key / pg.tok];
        return (which == 0 ? kc : vc) + pid * pg.page + (static_cast<int64_t>(key % pg.tok) * hk + kh) * D;
    };
    const int c0 = blockIdx.z * NT % D;
    slow_head<D>(q + (r * h + kh * G + hd) * D, kv, win > 0 ? max(p - win + 1, 0) : 0, p, (double)scale, oh, c0,
                 c0 + NT);
}

// merge_pg_odd over V-only pages (TF_GEMMA4_V_ONLY): a warp's keys from their value rows (vk_row).
template <int D>
__device__ __noinline__ void tf_wrows::merge_pg_odd_vo(const __nv_bfloat16* q, const __nv_bfloat16* vc, const int* pos,
                                                       const int* slot, int h, int hk, int win, float scale, Pg pg,
                                                       Vk vk, __nv_bfloat16* oh, int64_t r, int kh, int hd) {
    __shared__ __nv_bfloat16 kr[NW][D];
    const int p = pos[r], G = h / hk, ts = (slot != nullptr ? slot[r] : 0) * pg.stride;
    const auto kv = [&](int key, int which) -> const __nv_bfloat16* {
        const int64_t pid = pg.tab[ts + key / pg.tok];
        const __nv_bfloat16* row = vc + pid * pg.page + (static_cast<int64_t>(key % pg.tok) * hk + kh) * D;
        return which == 0 ? vk_row<D>(row, key, vk, kr[threadIdx.x / 32]) : row;
    };
    const int c0 = blockIdx.z * NT % D;
    slow_head<D>(q + (r * h + kh * G + hd) * D, kv, win > 0 ? max(p - win + 1, 0) : 0, p, (double)scale, oh, c0,
                 c0 + NT);
}

// prompt::odd over a stream's V-only pages (TF_GEMMA4_V_ONLY): a warp's keys from their value rows (vk_row).
template <int D>
__device__ __noinline__ void tf_wrows::prompt::odd_vo(const __nv_bfloat16* q, const __nv_bfloat16* vc, __nv_bfloat16* out, const int* hi,
                                    int p0, int W, int H, int HK, int G, float scale, int win, Pg pg, Vk vk) {
    using K = Cfg<D>;
    __shared__ __nv_bfloat16 kr[NW][D];
    const int HB = G % K::NG == 0 ? K::NG : 1, RB = K::NG / HB;
    const int hk = blockIdx.y / (G / HB), hg = blockIdx.y % (G / HB);
    const int row0 = blockIdx.x * 16 * RB;
    const auto kv = [&](int key, int which) -> const __nv_bfloat16* {
        const __nv_bfloat16* row = vc + (int64_t)pg.tab[key / pg.tok] * pg.page + ((int64_t)(key % pg.tok) * HK + hk) * D;
        return which == 0 ? vk_row<D>(row, key, vk, kr[threadIdx.x / 32]) : row;
    };
    for (int i = 0; i < 16 * RB * HB; ++i) {
        const int r = row0 + i % (16 * RB), head = hk * G + hg * HB + i / (16 * RB);
        if (r >= W) continue;
        __nv_bfloat16* dst = out + ((int64_t)r * H + head) * D;
        const bool affected = tf_wrows_head_affected(reinterpret_cast<const uint16_t*>(dst), D, threadIdx.x, NT);
        if (!__syncthreads_or(affected)) continue;
        const int p = p0 + r;
        slow_head<D>(q + ((int64_t)r * H + head) * D, kv, win > 0 ? max(p - win + 1, 0) : 0,
                     hi != nullptr ? hi[r] : p, (double)scale, dst, 0, D);
    }
}

// Dory's full-attention caches in pages (head dim 256, no window): kc/vc a layer's part of page 0, tab a row of ids.
extern "C" __global__ void __launch_bounds__(NT) tf_wrows_part_256_pg(
    const __nv_bfloat16* q, const __nv_bfloat16* kc, const __nv_bfloat16* vc, const int* pos, const int* slot,
    float* po, float* pm, float* pl, int h, int hk, int win, long long ring, float scale, const int* tab,
    long long page, int tok, int stride) {
    part<256, NONE, true>(q, kc, vc, pos, slot, po, pm, pl, h, hk, win, ring, scale, Pg{tab, page, tok, stride});
}

extern "C" __global__ void __launch_bounds__(NT) tf_wrows_merge_256_pg(
    const float* po, const float* pm, const float* pl, __nv_bfloat16* out, int h, int hk, int chunks,
    const __nv_bfloat16* q, const __nv_bfloat16* kc, const __nv_bfloat16* vc, const int* pos, const int* slot, int win,
    long long ring, float scale, const int* tab, long long page, int tok, int stride) {
    merge<256, NONE, true>(po, pm, pl, out, h, hk, chunks, q, kc, vc, pos, slot, win, ring, scale,
                           Pg{tab, page, tok, stride});
}

// The prompt kernel's tab is its stream's own row of page ids.
extern "C" __global__ void __launch_bounds__(NT, 1) tf_wrows_prompt_256_pg(
    const __nv_bfloat16* q, const __nv_bfloat16* kc, const __nv_bfloat16* vc, __nv_bfloat16* out, const int* hi,
    int p0, int W, int H, int HK, int G, float scale, int win, int ring, const int* tab, long long page, int tok) {
    prompt::kernel<256, false, NONE, true>(q, kc, vc, out, hi, p0, W, H, HK, G, scale, win, ring,
                                           Pg{tab, page, tok, 0});
}

// Spark2.5's full-attention prompt rows over pages (head dim 256, P32): tab the stream's row, tok % 32 == 0.
extern "C" __global__ void __launch_bounds__(NT, 1) tf_wrows_prompt_256_p32_pg(
    const __nv_bfloat16* q, const __nv_bfloat16* kc, const __nv_bfloat16* vc, __nv_bfloat16* out, const int* hi,
    int p0, int W, int H, int HK, int G, float scale, int win, int ring, const int* tab, long long page, int tok) {
    prompt::kernel<256, true, NONE, true>(q, kc, vc, out, hi, p0, W, H, HK, G, scale, win, ring,
                                          Pg{tab, page, tok, 0});
}

// Gemma 4's global decode over pages (head dim 512, no window): each key's page from tab, any tok; merge's replay too.
extern "C" __global__ void __launch_bounds__(NT) tf_wrows_part_512_pg(
    const __nv_bfloat16* q, const __nv_bfloat16* kc, const __nv_bfloat16* vc, const int* pos, const int* slot,
    float* po, float* pm, float* pl, int h, int hk, int win, long long ring, float scale, const int* tab,
    long long page, int tok, int stride) {
    part_staged<512, NONE>(q, kc, vc, pos, slot, po, pm, pl, h, hk, win, ring, scale, Vk{}, Pg{tab, page, tok, stride});
}

extern "C" __global__ void __launch_bounds__(NT) tf_wrows_merge_512_pg(
    const float* po, const float* pm, const float* pl, __nv_bfloat16* out, int h, int hk, int chunks,
    const __nv_bfloat16* q, const __nv_bfloat16* kc, const __nv_bfloat16* vc, const int* pos, const int* slot, int win,
    long long ring, float scale, const int* tab, long long page, int tok, int stride) {
    merge<512, NONE, true>(po, pm, pl, out, h, hk, chunks, q, kc, vc, pos, slot, win, ring, scale,
                           Pg{tab, page, tok, stride});
}

// Gemma 4's global prompt rows over pages: tab the stream's row; a 32-key tile takes one page base, so tok % 32 == 0.
extern "C" __global__ void __launch_bounds__(NT, 1) tf_wrows_prompt_512_pg(
    const __nv_bfloat16* q, const __nv_bfloat16* kc, const __nv_bfloat16* vc, __nv_bfloat16* out, const int* hi,
    int p0, int W, int H, int HK, int G, float scale, int win, int ring, const int* tab, long long page, int tok) {
    prompt::kernel<512, false, NONE, true>(q, kc, vc, out, hi, p0, W, H, HK, G, scale, win, ring,
                                           Pg{tab, page, tok, 0});
}

extern "C" __global__ void __launch_bounds__(NT, 1) tf_wrows_prompt_512_p32_pg(
    const __nv_bfloat16* q, const __nv_bfloat16* kc, const __nv_bfloat16* vc, __nv_bfloat16* out, const int* hi,
    int p0, int W, int H, int HK, int G, float scale, int win, int ring, const int* tab, long long page, int tok) {
    prompt::kernel<512, true, NONE, true>(q, kc, vc, out, hi, p0, W, H, HK, G, scale, win, ring,
                                          Pg{tab, page, tok, 0});
}

// Rows k, v [n, row] (row = hk * D elements) into the pages at their positions: a block a row, 16 bytes a thread.
extern "C" __global__ void __launch_bounds__(128) tf_wrows_put(const __nv_bfloat16* k, const __nv_bfloat16* v,
                                                               __nv_bfloat16* kc, __nv_bfloat16* vc, const int* pos,
                                                               const int* slot, const int* tab, long long page,
                                                               int tok, int stride, int row) {
    const int64_t r = blockIdx.x;
    const int p = pos[r];
    const int64_t id = tab[slot[r] * stride + p / tok];
    const int64_t at = id * page + static_cast<int64_t>(p % tok) * row;
    for (int e = threadIdx.x * 8; e < row; e += 128 * 8) {
        *reinterpret_cast<uint4*>(kc + at + e) = *reinterpret_cast<const uint4*>(k + r * row + e);
        *reinterpret_cast<uint4*>(vc + at + e) = *reinterpret_cast<const uint4*>(v + r * row + e);
    }
}

// TF_GEMMA4_V_ONLY: Gemma 4's global layers over V-only pages; kc is not read, each key comes from its value (vk_pair).
extern "C" __global__ void __launch_bounds__(NT) tf_wrows_part_512_pg_vo(
    const __nv_bfloat16* q, const __nv_bfloat16* kc, const __nv_bfloat16* vc, const int* pos, const int* slot,
    float* po, float* pm, float* pl, int h, int hk, int win, long long ring, float scale, const int* tab,
    long long page, int tok, int stride, const __nv_bfloat16* kw, const float* inv, const float* cs, int rot) {
    part_staged<512, NONE, true>(q, kc, vc, pos, slot, po, pm, pl, h, hk, win, ring, scale, Vk{kw, inv, cs, rot},
                                 Pg{tab, page, tok, stride});
}

extern "C" __global__ void __launch_bounds__(NT) tf_wrows_merge_512_pg_vo(
    const float* po, const float* pm, const float* pl, __nv_bfloat16* out, int h, int hk, int chunks,
    const __nv_bfloat16* q, const __nv_bfloat16* kc, const __nv_bfloat16* vc, const int* pos, const int* slot, int win,
    long long ring, float scale, const int* tab, long long page, int tok, int stride, const __nv_bfloat16* kw,
    const float* inv, const float* cs, int rot) {
    merge<512, NONE, true, true>(po, pm, pl, out, h, hk, chunks, q, kc, vc, pos, slot, win, ring, scale,
                                 Pg{tab, page, tok, stride}, Vk{kw, inv, cs, rot});
}

extern "C" __global__ void __launch_bounds__(NT, 1) tf_wrows_prompt_512_pg_vo(
    const __nv_bfloat16* q, const __nv_bfloat16* kc, const __nv_bfloat16* vc, __nv_bfloat16* out, const int* hi,
    int p0, int W, int H, int HK, int G, float scale, int win, int ring, const int* tab, long long page, int tok,
    const __nv_bfloat16* kw, const float* inv, const float* cs, int rot) {
    prompt::kernel<512, false, NONE, true, true>(q, kc, vc, out, hi, p0, W, H, HK, G, scale, win, ring,
                                                 Pg{tab, page, tok, 0}, Vk{kw, inv, cs, rot});
}

extern "C" __global__ void __launch_bounds__(NT, 1) tf_wrows_prompt_512_p32_pg_vo(
    const __nv_bfloat16* q, const __nv_bfloat16* kc, const __nv_bfloat16* vc, __nv_bfloat16* out, const int* hi,
    int p0, int W, int H, int HK, int G, float scale, int win, int ring, const int* tab, long long page, int tok,
    const __nv_bfloat16* kw, const float* inv, const float* cs, int rot) {
    prompt::kernel<512, true, NONE, true, true>(q, kc, vc, out, hi, p0, W, H, HK, G, scale, win, ring,
                                                Pg{tab, page, tok, 0}, Vk{kw, inv, cs, rot});
}

// Rows v [n, row] into the V pages only (tf_wrows_put's v half).
extern "C" __global__ void __launch_bounds__(128) tf_wrows_put_v(const __nv_bfloat16* v, __nv_bfloat16* vc,
                                                                 const int* pos, const int* slot, const int* tab,
                                                                 long long page, int tok, int stride, int row) {
    const int64_t r = blockIdx.x;
    const int p = pos[r];
    const int64_t id = tab[slot[r] * stride + p / tok];
    const int64_t at = id * page + static_cast<int64_t>(p % tok) * row;
    for (int e = threadIdx.x * 8; e < row; e += 128 * 8)
        *reinterpret_cast<uint4*>(vc + at + e) = *reinterpret_cast<const uint4*>(v + r * row + e);
}

// K from rows v vs the k rows qk_round wrote: counts += values compared, values differing; plant flips one bit.
extern "C" __global__ void __launch_bounds__(256) tf_vonly_check(const __nv_bfloat16* k, const __nv_bfloat16* v,
                                                                 const int* pos, int hk, const __nv_bfloat16* kw,
                                                                 const float* inv, const float* cs, int rot, int plant,
                                                                 unsigned long long* counts) {
    constexpr int D = 512;
    const int64_t r = blockIdx.x;
    const int kh = blockIdx.y, d = threadIdx.x;
    const int64_t at = (r * hk + kh) * D;
    __nv_bfloat16 e[2] = {v[at + d], v[at + d + D / 2]};
    vk_pair<D>(&e[0], &e[1], d, pos[r], Vk{kw, inv, cs, rot});
    uint16_t a0 = __bfloat16_as_ushort(e[0]);
    const uint16_t a1 = __bfloat16_as_ushort(e[1]);
    if (plant && r == 0 && kh == 0 && d == 3) a0 ^= 1;
    const int bad = (a0 != __bfloat16_as_ushort(k[at + d])) + (a1 != __bfloat16_as_ushort(k[at + d + D / 2]));
    if (bad) atomicAdd(&counts[1], static_cast<unsigned long long>(bad));
    if (threadIdx.x == 0) atomicAdd(&counts[0], static_cast<unsigned long long>(D));
}

// V-only keys' cos, then sin, of each position's rot rotated pairs (a block a position), shared by the global layers.
extern "C" __global__ void __launch_bounds__(128) tf_vonly_rope_table(const float* inv, int rot, float* out) {
    const int64_t p = blockIdx.x;
    for (int j = threadIdx.x; j < rot; j += blockDim.x) {
        const float a = __fmul_rn(static_cast<float>(p), inv[j]); // qk_rows.cuh rope_at's angle, then cosf and sinf
        out[p * 2 * rot + j] = cosf(a);
        out[p * 2 * rot + rot + j] = sinf(a);
    }
}
