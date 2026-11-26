// Qwen3.5-2B's keys and attention over core-pool pages: the contiguous kernels with only the K/V address changed.

kernel void qwen35_keys_paged(const device bfloat* k [[buffer(0)]],
                              const device bfloat* v [[buffer(1)]],
                              device uchar* keys [[buffer(2)]],
                              device uchar* values [[buffer(3)]],
                              constant uint4& dims [[buffer(4)]],
                              device const uint* table [[buffer(5)]],
                              constant ulong& page_bytes [[buffer(6)]],
                              uint3 p [[thread_position_in_grid]]) {
  const uint d = p.x, h = p.y, r = p.z;
  const uint m = dims.z + r;
  const ulong at = tf_paged_row(table, dims.x + r, page_bytes, h, 256);
  ((device bfloat*)(keys + at))[d] = qwen35_rotate(k + (size_t(m) * 2 + h) * 256, d, dims.x + r);
  ((device bfloat*)(values + at))[d] = v[(size_t(m) * 2 + h) * 256 + d];
}

// The chunk sits inside one page: its keys' and values' head rows there, from one table entry.
#define TF_Q35_KV const device uchar*
#define TF_Q35_EXTRA device const uint* table [[buffer(8)]], constant ulong& page_bytes [[buffer(9)]],
#define TF_Q35_CAP
#define TF_Q35_CHECK static_assert(TF_PAGE_TOKENS % CK == 0, "a chunk of keys must sit inside one page");
#define TF_Q35_CHUNK \
  const ulong at = ulong(table[k0 / TF_PAGE_TOKENS]) * page_bytes + ulong(h) * TF_PAGE_TOKENS * D * 2; \
  const device bfloat* kp = (const device bfloat*)(K + at); \
  const device bfloat* vp = (const device bfloat*)(V + at);
#define TF_Q35_ROW(KV, at, row) at + (row % TF_PAGE_TOKENS) * D + int(lane) * DPL
TF_Q35_ATTN_PARTIAL(qwen35_attn_partial_paged)
