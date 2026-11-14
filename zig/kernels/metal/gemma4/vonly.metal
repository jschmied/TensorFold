// Gemma 4 V-only global layers (keys are values): each key rebuilt from its cached value as gemma_norm_rope makes it.

// Key dims lo and lo + DH / 2 at position pos from value row v, in gemma_norm_rope's ops: bf16(KW * v) on both, roped.
inline void gemma_key_pair(const device bfloat* v, const device bfloat* KW, const device float* INVF, int lo, float pos,
                           thread bfloat& a, thread bfloat& b) {
  constexpr int HALF = DH / 2;
#ifndef TF_KEY_PLANT
  const float x1 = float(bfloat(float(KW[lo]) * float(v[lo])));
#else // the planted control: one product left unrounded
  const float x1 = float(KW[lo]) * float(v[lo]);
#endif
  const float x2 = float(bfloat(float(KW[lo + HALF]) * float(v[lo + HALF])));
  const float theta = pos * INVF[lo];
  const float c = metal::fast::cos(theta), s = metal::fast::sin(theta);
  a = bfloat(x1 * c - x2 * s);
  b = bfloat(x1 * s + x2 * c);
}

// Keys [HK, n, DH] from values [HK, n, DH]: row p at POS[p] (P.y) or at p; a simdgroup a row, lanes every 32nd pair.
[[kernel]] void gemma_key_rebuild(const device bfloat* V [[buffer(0)]], const device bfloat* KW [[buffer(1)]],
                                  const device float* INVF [[buffer(2)]], const device int* POS [[buffer(3)]],
                                  constant int2& P [[buffer(4)]], device bfloat* K [[buffer(5)]],
                                  uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  const int p = int(tg.x), h = int(tg.y), n = P.x;
  const device bfloat* v = V + (long(h) * n + p) * DH;
  device bfloat* k = K + (long(h) * n + p) * DH;
  const float pos = float(P.y ? POS[p] : p);
  for (int lo = int(lane); lo < DH / 2; lo += 32) {
    bfloat a, b;
    gemma_key_pair(v, KW, INVF, lo, pos, a, b);
    k[lo] = a;
    k[lo + DH / 2] = b;
  }
}

// A key block's 32 keys at positions kpos.. into kt [32][DH] by n threads, each 8 pairs a step (coalesced value reads).
inline void gemma_key_tile(threadgroup bfloat* kt, const device bfloat* vh, const device bfloat* KW,
                           const device float* INVF, int kpos, uint t, uint n) {
  constexpr int HALF = DH / 2, CH = HALF / 8;
  for (uint i = t; i < 32 * CH; i += n) {
    const int u = int(i) / CH, d0 = (int(i) % CH) * 8;
    const device bfloat* v = vh + long(kpos + u) * DH;
    for (int j = 0; j < 8; j++) {
      bfloat a, b;
      gemma_key_pair(v, KW, INVF, d0 + j, float(kpos + u), a, b);
      kt[u * DH + d0 + j] = a;
      kt[u * DH + d0 + j + HALF] = b;
    }
  }
}
