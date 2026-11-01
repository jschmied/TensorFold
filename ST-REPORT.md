# ST-CORE report: hand-written safetensors header readers against `core/safetensors.zig`

Branch `core-safetensors-host`, from upstream main `9e8c368b0`. Delete this file before any PR.

## Summary

- **Readers:** 28 places in 24 files read a safetensors header themselves.
  - 27 were ported into a differential fuzz harness and run against core on the same inputs; the 28th, `pull_exlr8_plan.headerOk`, was checked on its own.
  - Inputs: the corpus (208 files: 100 repo fixtures, 19 synthesized real layouts, 89 malformed and edge cases), 2.5 million mutants a copy, plus a second corpus of 17 MoE and h3-targeted files with 0.5 million mutants a copy.
  - Python `safetensors` 0.8.0 gave a third opinion on the corpus.
- **5 copies have defects** (each crash below reproduced in a Debug build; the downstream effects are from reading the code and were not run):
  - **Panics on malformed headers:** `xpu/loader.zig`, `glm/dflash_load.zig`, `flashnext/cuda_embed.zig`, `server/slide_model.zig`, `cli/pull_exlr8_plan.zig`. All are reachable from a file on disk.
  - **Allocation sized by the file:** `xpu/loader.zig` allocates the header length word before any check (up to 2^64−1 requested from a 1 KB file). `slide_model.zig` allocates `rows × cols × 2` from a shape (64 GiB requested from a 200-byte file).
  - **Out-of-bounds device reads or writes downstream:**
    - `cuda_embed.zig` (one of the 5) does not check bytes == dtype × shape, so its GPU table can be indexed past its buffer.
    - The looser `gemma4/xpu_dev.zig` has the same gap, so `xpu_load` copies more bytes than the device buffer it sized from the shape.
- **One copy gives wrong results on valid files:** `h3/checkpoint.zig` finds tensors by text search. A dtype found as a substring, or a match inside `__metadata__`, gives the wrong tensor. Valid JSON with spaces is refused.
- **The other 22 readers** (no crash found; a reader with mixed results is counted once, by its main class):
  - 5 are identical to core, because they already call it.
  - 4 are identical in result but do their own length step before core's parse. 2 of them are migrated (below).
  - 2 are looser without a design reason (`gemma4/xpu_dev.zig`, `xpu/safetensors.zig`).
  - 5 are looser by design or hash-only.
  - 6 are stricter on purpose.
- **Migrated: 2 copies, one commit each** (`core/eaq_container.zig`, `deepseek_v41/pack.zig`). Both were identical to core on every input and now read the length with `headerSize`.
  - Two more identical copies (`qwen_vl/weights.zig`, `h3/tensors.zig`) are Metal-only and not built here.
  - Nothing else was changed.
## Inventory

**How I found the readers:** I looked for every file with `readInt(u64` plus `data_offsets`, `__metadata__` or `.safetensors` (34 files), and every file with `data_offsets` but no u64 read (24 files: writers and fixtures). I read them all.

**What the 34 files hold:**
- 28 header readers in 24 files. I ported 27 into the fuzz harness and checked `pull_exlr8_plan` on its own (below).
- 4 test helpers that read only the data start of images they built themselves:
  - `deepseek_v41/{stage,engram_manifest_test,engram_pack_test,world4_stage_test}.zig`;
  - `tests/kimi_k3/validate.zig:39-40`. This one runs after `cluster.checkpoint.load` has checked the same file; see the defects.
- No header reader at all:
  - `core/quant/convert.zig:47`, `core/exlr8_dense.zig:189` and `exlr8_experts.zig:194` read u64 data words, not header lengths;
  - `flashnext/cuda_ple.zig` uses core's `File.open`;
  - in `flashnext/cuda_rank_engine.zig` the u64 reads are wire payloads;
  - `zig/safetensors_header_host.zig` is the core-against-cluster test.

**Column meanings** in the table below:
- **Reads:** *full* = every tensor; *one* = a named tensor; *length* = only the header length (the JSON is hashed or read for metadata).
- **Bytes:** *slice* = the caller passes the mapped or loaded file.
- **Core:** what it already calls in `core/safetensors.zig`.

| # | copy (file:lines) | reads | feeds | bytes | core | class (all corpus files + mutants) |
|---|---|---|---|---|---|---|
| 1 | `xpu/loader.zig:84-108` `addShard` | full | nemotron XPU weights | pread | none | **defect**: allocation sized by the file's length word; 12 panic kinds |
| 2 | `glm/dflash_load.zig:125-142` (in `load`) | full | GLM DFlash2 drafter weights (CUDA) | pread, counts ignored | none | **defect**: panics; parses unread heap bytes; offsets unchecked |
| 3 | `flashnext/cuda_embed.zig:15-61` `tensorAt`/`readTriple`/`open` | one (×3) | FlashNext CUDA embedding table | mmap | `DType.parse` only | **defect**: panics; no size == dtype × shape |
| 4 | `server/slide_model.zig:66-127` `readShard`/`offsets`/`inUse` | experts only | server's Living Weights view | pread | none | **defect**: overflow and narrowing panics, allocation sized by a shape |
| 5 | `cli/pull_exlr8_plan.zig:148-170` `headerOk` | metadata only | `tensorfold pull`: refetch or keep a width file's header run | pread of 65,536 B | none | **defect**: `n + 8` overflow panic (safe builds); accepts any tensor entries |
| 6 | `gemma4/xpu_dev.zig:25-54` `parseShard` | full | gemma4 XPU weights | mmap | none | **looser**: no byte-size, overlap or dtype check, rank ≤ 8 (agent reading: device-buffer overrun downstream) |
| 7 | `h3/checkpoint.zig:15-85` `locate`/`openWith`/`bf16` | one, by text search | h3 CUDA weights | mmap | none | **wrong result** on valid files, and looser |
| 8 | `xpu/safetensors.zig:26-62` `parse` | full | XPU weights (`xpu/dev.zig`) | mmap (slice) | none | **looser**: rank 6-8, overlaps, no cap. **Stricter**: no F8_E8M0 |
| 9 | `core/checkpoint_identity.zig:209-231` `read` | ranges | split checkpoint content digest | read | `headerSize` | **looser**: no dtype/shape/size check (hash only). **Stricter**: an empty range inside another |
| 10 | `core/checkpoint_identity.zig:49-57` `header` | length | identity hash | pread | `headerSize` | identical |
| 11 | `core/checkpoint_metal.zig:76-90` `addFileSplit` | full | Metal weights | pread | `headerSize` + `parseHeaderPrefix` | identical |
| 12 | `core/drafter_layout.zig:95-106` `readHeader` | full | drafter release check | pread | `parseHeader` | **stricter**: header cap 64 MiB |
| 13 | `core/eaq_container.zig:303-325` `readShard` | full | EAQ container open | pread | `parseHeader` | identical, cap 256 MiB before core's 100 MB → **migrated** |
| 14 | `core/exlr8_experts.zig:224-234` `metadata` | metadata | EXLR8 widths | mmap (after `File.open`) | `File.open` | **stricter**: a non-object `__metadata__` |
| 15 | `core/qwen_vl/weights.zig:153-161` `headers` | full (prefix) | Qwen-VL tower weights (Metal) | pread | `parseHeaderPrefix` | identical, no length cap (allocation up to the file size); Metal-only |
| 16 | `cluster/checkpoint.zig:79-98` `parseHeaderInFile` | full | cluster planner | slice | `parseHeader` | **stricter**: rank 5 refused (cluster `max_rank = 4`) |
| 17 | `cluster/checkpoint.zig:131-145` `load` | length, then #16 | cluster planner | pread | via #16 | as #16; allocates up to 1 GiB before core's cap |
| 18 | `deepseek_v41/pack.zig:91-98` `Pack.open` | length, then `addHeader` | DSv4.1 pack, plan, load | pread | `parseHeader` | identical, cap 1 GiB before core's 100 MB → **migrated** |
| 19 | `deepseek_v41/pack.zig:158-163` `addHeaderDump` | full | header dumps (no data) | slice | `parseHeader` | **looser** by design: data length is `maxInt(u64)`, so end ≤ data is not checked |
| 20 | `flashnext/pack_io.zig:143-161` `headerLen`/`identityFromMapped` | length | pack source identity hash | mmap | none | **looser**: no 100 MB cap (hash only) |
| 21 | `flashnext/pack_io.zig:343-372` `packReady` | full | reuse a pack cache or rebuild it | read | `parseHeader` | **stricter**: cap 64 MiB, length 0 refused, tensors must end at EOF, no empty header |
| 22 | `flashnext/replay_run.zig:208-228` `indexFile` | full | FlashNext Metal replay | pread | `headerSize` + `parseHeader` | identical |
| 23 | `glm/names.zig:53-62` `read` | full (prefixes) | GLM Metal weight map | pread | `headerSize` + `parseHeaderPrefixes` | identical |
| 24 | `glm/vision.zig:410-420` `loadWeights` | full (prefix) | GLM Metal vision weights | pread | `headerSize` + `parseHeaderPrefix` | identical |
| 25 | `h3/tensors.zig:14-22` `open` | full | h3 Metal weights | mmap | `parseHeader` | identical (core's JSON cap applies); Metal-only |
| 26 | `kimi_k3/shard_header.zig:15-41` `parse` | full (prefix) | Kimi K3 Metal zero-copy | slice | `headerSize` + `parseHeaderPrefix` | **stricter** by design (8-byte alignment, rank ≤ 3, dtype bf16/f32/u8); **looser** only outside its `language_model.` namespace (core's prefix option) |
| 27 | `qwen3_5/weights/shard.zig:17-29,56-58` `open`/`parse` | full | Qwen3.5 weights | mmap | `headerSize` + `parseHeaderOptions(.omit)` | **looser** by design: entries with an unnamed dtype or rank > 5 are left out unchecked (core's `.omit` option) |
| 28 | `hip/comm/identity.zig:75-86` `shardDigest` | length | HIP rank-join identity hash | pread | none | **looser**: cap 1 GiB, not 100 MB (hash only) |

## Fuzz setup

The harness is a scratch program outside the tree (`/home/user/scratch/st/h`, not committed).

**Copies.** Each copy is ported as written, with these changes:
- A positional read becomes a read from the bytes in memory. It still returns a short count past the end.
- `xpu/safetensors.zig` is compiled unchanged: it only uses `std`.
- Core's `safetensors.zig` is compiled unchanged. The reference is `headerSize`, then `parseHeader` over the data length that is left, exactly as `File.open` reads.
- Where a copy indexes only part of the header by design, core's answer is filtered the same way before comparing:
  - `identity.read` skips expert names;
  - `glm/dflash_load` takes BF16 only;
  - `h3` `locate` takes BF16 rank 1/2;
  - `kimi_k3` takes `language_model.` only.

**Assumptions** for copies that need more context than a header:
- `slide_model` uses the real `mlp.shared_expert.` family naming, pocket 1, and every layer as a slot.
- `h3` `locate` is asked for every top-level name of the header.
- `cuda_embed` is asked for every name, with the dtype the header gives it.
- `packReady` drops its `tf_source` identity step.
- `qwen_vl` uses prefix `""`.
- `glm_dflash`'s ignored reads leave `0xaa` in the unread bytes.

**What is compared:**
- accept or refuse;
- the data section's start;
- every tensor's name, dtype, shape and range relative to the data start (for range-only readers: names and ranges);
- for length-only readers: only the length step, against `headerSize`.

**Crash and allocation detection:**
- **Build:** ReleaseSafe, so bounds, overflow, `@intCast`, `.?` and union-field checks panic.
- **Isolation:** each copy runs in its own forked child, with a shared word holding the current input. When a child dies, the parent records that input and restarts the child at the next one.
- **Allocation:** every allocation goes through an allocator that refuses (and records) requests over 256 MiB. A request over 64 MiB and over 4× the file is reported.
- **Exact lines:** every panic was replayed in a Debug build for the exact line.

**Corpus:**
- **main, 208 files:**
  - 100 header literals from the repo's Zig sources and tests, each made a file whose data ends at the last tensor;
  - 19 synthesized layouts at hidden size 64 with real names: bf16 Llama-style; FP8 E4M3 with F32 or E8M0 block scales; MLX 4-bit with BF16 scales/biases; NVFP4 with E4M3 scales, `weight_scale_2` and `input_scale`; a rank-5 vision patch; rank 0 and empty tensors; every dtype; padded, unsorted, gapped and data-tail files; escaped and raw UTF-8 names;
  - 89 malformed cases (the 70 MB and 150 MB valid headers among them):
    - header length: 0, past the file, the cap ± 1, 2^32, 2^63, 2^64−1, 2^64−8, files under 8 bytes;
    - JSON: truncated, non-object, trailing bytes, BOM, invalid UTF-8, lone surrogates, duplicate names and fields, 5,000-deep nesting, 60,000 entries, a 3 MB string, 70 MB and 150 MB headers;
    - `__metadata__`: non-string values;
    - extra keys;
    - entries: end < begin, offsets past the data, negative, float, string, or wrong arity; byte size short or long; shape products that wrap `u64` to 0 or 4; rank 6/8/9/40; dimensions 2^64; unknown and lowercase dtypes; non-object entries;
    - overlapping (equal, partial, inside, an empty one inside) and gapped ranges.
- **extra, 17 files:** shared-expert MoE layouts and their overflow cases, plus h3 text-search cases.
- **mutants:** 2,500,000 per copy from the 63 valid bases of the main corpus, and 500,000 per copy from the extra corpus.
  - Each mutant applies 1-3 structure-aware edits: a number to an edge value (0, ±1, 2^31…2^64, 1e3, 1.0, `"4"`, `null`, `[]`), a dtype to another name (valid, unknown, lowercase, `F4`, `F6_E2M3`…), the length word (0, ±1, near 2^64, powers of two), a byte flip, a cut, data grown or shrunk, an entry duplicated (same or new name), a value dropped, trailing bytes, or a string replaced (empty, `__metadata__`, `\u0000`, `\ud800`, `0xff`).
  - Mutants are deterministic in their index, so a recorded index regenerates its input.
- **Throughput:** about 2,000 inputs/s per copy, and slower for copies that crash often (a fork per crash). The three crash-heavy copies ran on the main corpus without the files over 5 MB (206 files); those files only exercise the length caps, which are covered on the other copies.

**Third opinion:** Python `safetensors` 0.8.0, `deserialize(bytes)`, on both corpora. On the main corpus it agrees with core on 185 of 208 files.
- **Python accepts, core refuses (10):**
  - rank 6+ (core's `max_rank = 5`);
  - dtype `F4`;
  - `shape [16777216,16777216,32768,0]` (two fixtures): core refuses because the product overflows before the 0; Python's product is 0.
- **Python refuses, core accepts (13):**
  - gaps, a data tail, and an empty range inside another: Python wants the data covered exactly, with no gaps;
  - non-string `__metadata__` values and a 5,000-deep `__metadata__`: Python wants string values.
- On the extra corpus Python also refuses `h3_metadata_nesting`, for its object-valued metadata.

## Classes

**What the counts mean.** Counts are inputs where the copy disagrees with core:
- **crash:** the child died (a panic in ReleaseSafe);
- **looser:** the copy accepts what core refuses;
- **stricter:** the copy refuses what core accepts;
- **differs:** both accept, with other names, dtypes, shapes or ranges;
- **big alloc:** a request over 64 MiB and over 4× the file.

**Which runs:**
- "main" is the 208-file corpus plus 2.5 M mutants. For the three crash-heavy copies it is the 206 files under 5 MB, and for `h3_checkpoint.locate` it is a rerun that takes names from the parsed JSON.
- "extra" is the 17-file MoE / h3 corpus plus 0.5 M mutants. `slide_model`'s extra run uses the real `mlp.shared_expert.` naming; its main run had no shared-expert names to reach.

| copy | main: files + mutants run | crash | looser | stricter | differs | big alloc | extra: files + mutants run | crash | looser | stricter | differs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `checkpoint_metal.addFileSplit` | 2,500,208 | 0 | 0 | 0 | 0 | 0 | 500,012 | 0 | 0 | 0 | 0 |
| `cluster.load` | 2,500,208 | 0 | 0 | 5,475 | 0 | 0 | 500,012 | 0 | 0 | 0 | 0 |
| `cluster.parseHeader` | 2,500,208 | 0 | 0 | 5,475 | 0 | 0 | 500,012 | 0 | 0 | 0 | 0 |
| `cuda_embed.readTriple` | 2,500,196 | 237,350 | 124,785 | 18,002 | 0 | 1 | 500,008 | 51,380 | 27,889 | 3,347 | 0 |
| `drafter_layout.readHeader` | 2,500,208 | 0 | 0 | 1 | 0 | 0 | 500,012 | 0 | 0 | 0 | 0 |
| `dsv41_pack.addHeaderDump` | 2,500,208 | 0 | 62,504 | 0 | 0 | 0 | 500,012 | 0 | 12,275 | 0 | 0 |
| `dsv41_pack.open` | 2,500,208 | 0 | 0 | 0 | 0 | 0 | 500,012 | 0 | 0 | 0 | 0 |
| `eaq_container.readShard` | 2,500,208 | 0 | 0 | 0 | 0 | 0 | 500,012 | 0 | 0 | 0 | 0 |
| `exlr8_experts.metadata` | 2,500,208 | 0 | 0 | 16,231 | 0 | 0 | 500,012 | 0 | 0 | 0 | 0 |
| `flashnext_pack_io.identity` | 2,500,208 | 0 | 1 | 0 | 0 | 0 | 500,012 | 0 | 0 | 0 | 0 |
| `flashnext_pack_io.packReady` | 2,500,208 | 0 | 0 | 51,274 | 0 | 0 | 500,012 | 0 | 0 | 10,974 | 0 |
| `flashnext_replay_run.indexFile` | 2,500,208 | 0 | 0 | 0 | 0 | 0 | 500,012 | 0 | 0 | 0 | 0 |
| `gemma4_xpu_dev.parseShard` | 2,500,208 | 0 | 221,599 | 17 | 0 | 0 | 500,012 | 0 | 40,987 | 0 | 0 |
| `glm_dflash_load.load` | 2,500,151 | 67,755 | 45,133 | 115,785 | 0 | 0 | 500,008 | 26,784 | 19,946 | 16,154 | 0 |
| `glm_names.read` | 2,500,208 | 0 | 0 | 0 | 0 | 0 | 500,012 | 0 | 0 | 0 | 0 |
| `glm_vision.loadWeights` | 2,500,208 | 0 | 0 | 0 | 0 | 0 | 500,012 | 0 | 0 | 0 | 0 |
| `h3_checkpoint.locate` | 2,500,208 | 0 | 1,297,944 | 121,997 | 0 | 0 | 500,012 | 0 | 229,039 | 17,494 | 6,329 |
| `h3_tensors.open` | 2,500,208 | 0 | 0 | 0 | 0 | 0 | 500,012 | 0 | 0 | 0 | 0 |
| `hip_identity.shardDigest` | 2,500,208 | 0 | 1 | 0 | 0 | 0 | 500,012 | 0 | 0 | 0 | 0 |
| `identity.header` | 2,500,208 | 0 | 0 | 0 | 0 | 0 | 500,012 | 0 | 0 | 0 | 0 |
| `identity.read` | 2,500,208 | 0 | 287,331 | 3,932 | 0 | 0 | 500,012 | 0 | 76,983 | 0 | 0 |
| `kimi_k3.shard_header.parse` | 2,500,208 | 0 | 94,712 | 112,228 | 0 | 0 | 500,012 | 0 | 14,205 | 29,677 | 0 |
| `qwen3_5.shard.open` | 2,500,208 | 0 | 73,736 | 0 | 0 | 0 | 500,012 | 0 | 13,987 | 0 | 0 |
| `qwen_vl.headers` | 2,500,208 | 0 | 0 | 0 | 0 | 0 | 500,012 | 0 | 0 | 0 | 0 |
| `slide_model.readShard` | 2,500,208 | 0 | 0 | 1,295,662 | 0 | 42 | 500,017 | 9,804 | 0 | 257,015 | 0 |
| `xpu_loader.addShard` | 2,500,200 | 327,874 | 405,755 | 0 | 0 | 79,723 | 500,008 | 64,888 | 76,504 | 0 | 0 |
| `xpu_safetensors.parse` | 2,500,208 | 0 | 27 | 8,721 | 0 | 0 | 500,012 | 0 | 2 | 109 | 0 |

| class | copies | examples (header JSON; data follows) |
|---|---|---|
| **defect: crash** (D1-D5) | `xpu/loader`, `glm/dflash_load`, `flashnext/cuda_embed`, `server/slide_model`, `cli/pull_exlr8_plan` | `{"xBF16":{"dtype":"F16","shape":[1],"data_offsets":[0]}}` (index out of bounds); `{"w":{"dtype":"F16","shape":[1],"data_offsets":[-1,2]}}` (integer does not fit); `{"xBF16":{"dtype":"F16","shape":[],"":[0,2]}}` (null value) |
| **defect: allocation sized by the file** | `xpu/loader` (length word), `server/slide_model` (shape) | length word `0xffffffffffffffff` before a 1 KB header; `"shape":[2,34359738368]` on a `…shared_expert.up_proj.scales` entry |
| **defect: wrong result on a valid file** (D6) | `h3/checkpoint` `locate` | `{"x":{"dtype":"F16","shape":[1],"data_offsets":[0,2],"n":"BF16"}}` read as BF16; a `__metadata__` look-alike taken for the tensor |
| **looser, no design reason** | `gemma4/xpu_dev` (no size, dtype or overlap check; rank ≤ 8), `xpu/safetensors` (rank ≤ 8, overlaps, no cap) | `{"":{"dtype":"U8","shape":[],"data_offsets":[0,4]}}`; `{"w":{"dtype":"U8","shape":[1,1,1,1,1,1],"data_offsets":[0,1]}}`; two tensors on `[0,4]` |
| **looser by design** | `deepseek_v41/pack` `addHeaderDump`, `qwen3_5/weights/shard` (`.omit`), `checkpoint_identity` `read`, `flashnext/pack_io` identity, `hip/comm/identity` (hashes) | `{"x":{"dtype":"Q4","shape":[9],"w":[0,9]}}` (left out by `.omit`); a 150 MB header (hash only) |
| **stricter on purpose** | `cluster/checkpoint` (2), `core/drafter_layout`, `core/exlr8_experts` `metadata`, `flashnext/pack_io` `packReady`, `kimi_k3/shard_header` | rank 5; a 70 MB header; `{"__metadata__":[],…}`; `{}`; an unaligned header end |
| **identical, already core** | `checkpoint_identity` `header`, `checkpoint_metal`, `flashnext/replay_run`, `glm/names`, `glm/vision` | — |
| **identical, own length step** | `core/eaq_container` (**migrated**), `deepseek_v41/pack` `open` (**migrated**), `core/qwen_vl/weights`, `h3/tensors` | — |

Copies listed twice above have more than one class; the table gives each copy's every class. `xpu/safetensors` is also stricter on `F8_E8M0`, `kimi_k3` also looser outside its namespace, and `checkpoint_identity.read` also stricter on an empty range inside another.

## Defects

Notation used below:
- Example headers are the JSON after the 8-byte length word; the data section follows. All are among the shortest mutants or corpus files of their kind.
- "Debug line" is where a Debug build of the port panics; it maps one-to-one to the original line given.
- Release builds: the engines build with the `optimize` option (default fast). In ReleaseFast these panics are undefined behaviour instead of a stop.

### D1. `xpu/loader.zig:84-108` `Loader.addShard` (Nemotron-H XPU engine): allocation from the length word, 12 panic kinds

- **Allocation sized by the file.** Line 88 reads the length word and line 89 allocates it, with no bound and no check against the file. The harness saw 79,723 inputs whose request was over 64 MiB and over 4× the file, the largest 2^64−1 from a 1,006-byte file (`len = 0xffffffffffffffff`).
  - Effect: a `0xff…` word gives OutOfMemory.
  - A word below free memory but past the file allocates that much, then fails the read (`ReadFailed`).
  - From reading, not run: the fd opened at 85 is not closed on any error path.
- **Panics on malformed JSON:**

  | trigger (header) | panic | line |
  |---|---|---|
  | `[]` or a non-object root, e.g. `2"x":{…}` → `2` | access of union field `object` while `integer` is active | 94 |
  | an entry that is not an object, e.g. `{"a":{…},"Ztype":"F16",…}` | `object` while `string` | 97 |
  | no `data_offsets`: `{"x":{"dtype":"F16","shape":[]}}` | attempt to use null value | 98 |
  | `"data_offsets":[]` / `[0]` | index out of bounds | 99 / 100 |
  | `"data_offsets":[-1,2]` | integer does not fit in destination type | 99 |
  | `"data_offsets":[-0,2]`, `[18446744073709551616,2]`, `[true,2]`, `[null,2]`, `[[],2]`, `"4"` | access of union field `integer` while `float` / `number_string` / `bool` / `null` / `array` / `string` is active | 99, 100 |
  | `"data_offsets":[7,2]` (end before begin) | integer overflow (`hi - lo`) | 104 |
  | no `dtype`, or not a string | null value / union access | 105 |

- **Looser:** offsets are not checked against the file, dtype × shape is not checked, and overlaps are not checked. 405,755 + 76,504 accepted inputs that core refuses (main + extra run); 327,874 + 64,888 inputs crashed it.
  - From reading, not run: `load()` then casts `off + done` to `off_t` at line 23, which panics for offsets past 2^63. `loadF32` (161) allocates `len` from the header.

### D2. `glm/dflash_load.zig:125-142` (in `load`; GLM DFlash2 drafter to CUDA): panics, ignored read counts, offsets unchecked

- **Panics:**

  | trigger | panic | line |
  |---|---|---|
  | a non-object root (`0`, `[]`) | union access `object` | 136 |
  | an entry not an object | union access `object` | 139 |
  | no `dtype`: `{"xBF16":{"dt,pe":"F16",…}}` | null value | 139 |
  | `data_offsets` `[]` / `[ 2]` | index out of bounds | 141 |
  | `[8,2]` (end < begin) | integer does not fit (`@intCast` of a negative difference) | 141 |
  | non-integers (`-0`, a 2^63 number string, `true`, `null`, `"4"`, `[]`, `{}`) | union access `integer` | 141 |

- **Read counts ignored (lines 126, 131).** The two `readPositionalAll` results are discarded.
  - A file shorter than 8 bytes takes its length word from uninitialised stack bytes.
  - A length word up to 1 MiB past the end parses the unread rest of an uninitialised heap buffer as JSON.
  - In the port the unread bytes are `0xaa`, so that JSON fails; real memory may hold anything.
  - Line 50 also ignores the count when the tensor bytes are read.
- **Looser:**
  - Offsets are never checked against the file, and the shape is ignored. 45,133 + 19,946 accepted inputs that core refuses (67,755 + 26,784 crashed it), e.g. `{"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,999999]}}`.
  - The byte count is later checked against a constant per tensor (line 45). An entry with the right count but past EOF uploads whatever the buffer held before (from reading, not run).

### D3. `flashnext/cuda_embed.zig:15-61` `tensorAt` / `readTriple` / `Mapped.open` (FlashNext CUDA embedding table): panics, shape not checked against bytes

- **Panics** on any malformed field of the three tensors it reads (the harness asks for every name):

  | trigger | panic | line |
  |---|---|---|
  | root not an object | union access `object` | 34 |
  | entry not an object | union access `object` | 16 |
  | no `dtype` | null value | 17 |
  | no `shape` | null value | 19 |
  | no `data_offsets` | null value | 21 |
  | `[]` / `[0]` | index out of bounds | 22 / 23 |
  | `[-1,2]` | integer does not fit | 22 |
  | non-integer offsets or dimensions (`-0`, a 2^64 number string, `true`, `null`, `"4"`, `[]`, `{}`) | union access `integer` | 22, 23, 26 |

  In the harness, 237,350 of the 2.5 M main-run inputs and 51,380 of the 0.5 M extra-run inputs crashed this copy.
- **No size == dtype × shape check.** 124,785 + 27,889 accepted inputs that core refuses, e.g. `{"w":{"dtype":"U32","shape":[1000000,4],"data_offsets":[0,16]}}`.
  - From reading, not run: `Table.upload` (88-101) sizes the device buffers from the byte ranges but takes `rows` and `dims` from the shapes, so the embedding kernels index past the buffers.
- **No header cap:** a header as long as the file is parsed.

### D4. `server/slide_model.zig:66-127` `readShard` / `inUse` (server's Living Weights view): overflow and narrowing panics, allocation from a shape

The header's errors are caught (`catch {}` at 49), but panics are not. All were run on files with the real family's `mlp.shared_expert.` names (`living/families.zig:44`):

| trigger | effect | line |
|---|---|---|
| three shared-expert entries with `"data_offsets":[0,9223372036854775807]` | integer overflow (`expert_bytes +=`) | 84 |
| a name with layer 65536: `language_model.model.layers.65536.mlp.shared_expert.down_proj.weight` | integer does not fit (`@intCast` to u16) | 85 |
| `…shared_expert.up_proj.scales` with `"shape":[4611686018427387904,2]` | integer overflow (`own * row`) | 121 |
| `…shared_expert.up_proj.scales` with `"shape":[2,34359738368]` | allocates `pocket × cols × 2` = 64 GiB from a 200-byte file (the harness's allocator refused it) | 120 |

Offsets are never checked against the data. 9,804 of the extra corpus's 0.5 M inputs panicked this copy, and 42 made a request over 64 MiB.

### D5. `cli/pull_exlr8_plan.zig:148-170` `headerOk` (`tensorfold pull`): overflow, and corrupt headers kept

Checked on its own with 65,536-byte header runs, as `Scan.work` passes them:
- **Overflow.** A length word ≥ 2^64−8 makes `n + 8` (line 151) overflow. A Debug and a ReleaseSafe build of the port panic with "integer overflow"; ReleaseFast wraps and returns false.
- **Accepts corrupt tensor entries.** It checks only `__metadata__`. These header runs return true while core's `parseHeader` refuses them:
  - `{"__metadata__":{"format":"exlr8-v2","contents":"trellis","layer":"40","k":"3"},"t":0}`
  - the same with `"t":{"dtype":"U8","shape":[4],"data_offsets":[0,999999]}`
  - the same with `"t":{"dtype":"Q4","shape":[1,1,1,1,1,1,1],"data_offsets":[8,0]}`

  So the pull keeps such a run instead of refetching it. The later `Layer.open` (core `File.open`) refuses the file.

### D6. `h3/checkpoint.zig:15-85` `locate` (h3 CUDA weights): wrong tensor on valid files

It finds `"name":{` as text, takes the text up to the next `}`, and treats the tensor as BF16 if `"BF16"` appears anywhere in it. On valid files that core reads correctly:

| header (valid JSON) | core | `locate` |
|---|---|---|
| `{"x":{"dtype":"F16","shape":[1],"data_offsets":[0,2],"n":"BF16"}}` | `x` is F16 | `x` read as BF16 |
| `{"__metadata__":{"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}},"a":{"dtype":"BF16","shape":[1],"data_offsets":[2,4]}}` | `a` at [2,4] | `a` at [0,2] (the match inside `__metadata__`) |
| `{"a": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}` (spaces) | `a` | MissingTensor |

It also accepts what core refuses (1,297,944 + 229,039 inputs), e.g. truncated or invalid JSON whose matched text still parses, duplicate keys (the first wins), and trailing garbage. It refuses files under 16 bytes and rank 0 or > 2.

### Looser copies with a downstream effect

- **`gemma4/xpu_dev.zig:25-54` `parseShard`** (gemma4 XPU weights) has no byte-size == dtype × shape check, no dtype list, no overlap check, and allows rank ≤ 8. 221,599 + 40,987 accepted inputs that core refuses (main + extra run), e.g. `{"":{"dtype":"U8","shape":[],"data_offsets":[0,4]}}`.
  - From reading, not run: `xpu_load.zig` `projection8` / `projection` (60-69, 102-116) size device buffers from the shapes, then `Dev.copyIn` (202-208) writes `bytes.len` bytes at a raw device pointer with no bound. A tensor whose `data_offsets` span more than its shape implies is written past its device buffer.
- **`xpu/safetensors.zig:26-62` `parse`** (XPU weights through `xpu/dev.zig`) accepts rank 6-8 and overlapping tensors, and has no header cap.
  - Examples: `{"w":{"dtype":"U8","shape":[1,1,1,1,1,1],"data_offsets":[0,1]}}`; `overlap_partial`, `overlap_same_range` and `overlap_inside` in the corpus.
  - It refuses `F8_E8M0` (8,721 + 109 inputs).

## What was migrated

Two copies were identical to core on every input of both corpora and all mutants: same accept/refuse, same names, dtypes, shapes and ranges. They reach `core` without build changes, and they differed only in the length step. Each now calls `headerSize`; one commit per family.

| commit | copy | before | after |
|---|---|---|---|
| `core: eaq_container reads a shard's header length with safetensors.headerSize` | `core/eaq_container.zig` `readShard` | `n > 1 << 28 or n > len - 8` (own cap 256 MiB), then `alloc(n)`, then core `parseHeader` | `safetensors.headerSize(word, len) catch return badShard(...)`; the `max_header` constant is gone |
| `deepseek_v41: Pack.open reads a shard's header length with core's headerSize` | `deepseek_v41/pack.zig` `Pack.open` | `n > len - 8 or n > 1 << 30` (own cap 1 GiB), then `alloc(n)`, then core via `addHeader` | `try st.headerSize(word, len)` (same `error.BadSafetensors`) |

**What changes:**
- Results: none, because core's `parseHeader` already refused a header over 100 MB. What changes is how much is allocated and read before that refusal: at most 100 MB, not 256 MiB or 1 GiB.
- `headerSize` also checks `file_size < 8`, which the old `len - 8` did not, if a file shrinks between the two reads.

**Evidence:**
- **Fuzz.** On both corpora and 3 million mutants each, neither port disagrees with core: 0 lines in their result files, and no large allocation.
- **Tests kept in the tree:**
  - `core/eaq_container_test.zig` "a shard's header length is read as core reads it: one byte past the file or the u64 maximum is refused". It uses two new `eaq_fixture.zig` tampers (`header_past_file`, `header_max`). The valid case is the existing "a good pack verifies and exposes its entries".
  - `deepseek_v41/pack.zig` "a shard's header length is read as core reads it: past the file, past 100 MB or the u64 maximum is refused". It covers:
    - a valid shard (indexed, 12 bytes);
    - a length one byte past the file;
    - `max_header_bytes + 1` on a sparse file long enough to hold it;
    - 2^64−1 and 2^64−8, the values where `8 + n` would overflow.
- **Planted control** (on a scratch copy, not committed): with the eaq length check removed (`n = @intCast(word)`), the new eaq test fails (integer overflow panic in `len - 8 - n`).

## What was not migrated, and why

**Identical to core, but not built here.** These call core's parser already and differ only in the length step. Both are in the Metal engine only; no Linux step builds them, so I left them.
- `core/qwen_vl/weights.zig:153-161` has no length cap: it allocates up to the file size before core's 100 MB check refuses.
- `h3/tensors.zig:14-22` uses mmap, so it allocates nothing. It is missing `headerSize`, but core's JSON-length check gives the same result.

The change would be `st.headerSize(word, size)` in place of the hand-written check. Both files already import core's `safetensors.zig` (relative path, or the `safetensors` module).

**Already core:**
- `checkpoint_identity.header`, `checkpoint_metal.addFileSplit`, `flashnext/replay_run.indexFile`, `glm/names.read` and `glm/vision.loadWeights` call `headerSize` and `parseHeader*`.
- `kimi_k3/shard_header.parse` and `qwen3_5/weights/shard` do too, plus their own deliberate rules.

There is nothing to migrate in any of them.

**Stricter on purpose** (changing them would change what loads):

| copy | refuses beyond core | example |
|---|---|---|
| `cluster/checkpoint.zig` `parseHeaderInFile` / `load` | rank 5 (cluster `max_rank = 4`) | the `layout_vision_rank5` corpus file |
| `core/drafter_layout.zig` `readHeader` | headers of 64-100 MB | the 70 MB corpus file |
| `core/exlr8_experts.zig` `metadata` | a non-object `__metadata__` | `{"__metadata__":[],"w":{…}}` |
| `flashnext/pack_io.zig` `packReady` | length 0; > 64 MiB; tensors that do not end at EOF; no tensors | `{}`; `layout_data_tail` |
| `kimi_k3/shard_header.zig` `parse` | header end or tensor not 8-byte aligned; rank > 3; dtype other than bf16/f32/u8 | most fixtures (`MisalignedShard`) |

**Looser by design:**

| copy | accepts beyond core | why |
|---|---|---|
| `deepseek_v41/pack.zig` `addHeaderDump` | ranges past the (absent) data | a header dump has no data; `data_len = maxInt(u64)` |
| `qwen3_5/weights/shard.zig` | entries with an unnamed dtype or rank > 5, left out unchecked | core's `.omit` option |
| `kimi_k3/shard_header.zig` | anything outside `language_model.` | core's prefix option |
| `checkpoint_identity.read`, `flashnext/pack_io` identity, `hip/comm/identity.zig` `shardDigest` | malformed JSON (hashed as bytes), and headers up to the file size or 1 GiB | identity hashes; a later load parses with core |

`checkpoint_identity.read` is also stricter in one case: an empty range inside another tensor (`overlap_empty_inside`), which core skips in its overlap check.

**Defective or looser without a reason** (D1-D6, plus `gemma4/xpu_dev` and `xpu/safetensors`): not changed, because the brief keeps their behaviour. The change that fits each:
- `xpu/loader.zig`, `glm/dflash_load.zig`, `flashnext/cuda_embed.zig`, `gemma4/xpu_dev.zig`, `xpu/safetensors.zig`: `headerSize` + `parseHeader` (or `parseHeaderOptions`), then their own dtype and name rules on the entries.
  - `cuda_embed`, `glm/dflash_load` (module `glm/cuda.zig`, `cuda_engines.zig:7`) and `gemma4/xpu_dev` (`xpu.zig:99,142`) are in modules that import `core`.
  - `xpu/loader.zig` and `xpu/safetensors.zig` are not: see the module-reach list.
- `server/slide_model.zig`: core's parse for the header, `std.math.add` / `mul` and a `std.math.cast` for the layer, and a size bound before `inUse` allocates.
- `cli/pull_exlr8_plan.zig`: `n != header_bytes - 8`, plus `exlr8.safetensors.parseHeader(arena, bytes[8..], size - header_bytes)` so a corrupt run is refetched.
- `h3/checkpoint.zig`: parse the header with core and look the name up in the index.

**Module reach** (files that cannot import `core` without a `build.zig` change). I checked the imports each module is created with:

| file | module | change it would take |
|---|---|---|
| `xpu/loader.zig` | `xpu/root.zig` module, which imports only `ze` (`zig/build/xpu.zig:192`) | add `core` to that module |
| `xpu/safetensors.zig` | `xpu/qwen27.zig` (`xpu.zig:133`, imports `xpu_qwen27_kernels` and `ze`) and its own test module; only `qwen27_main` (`xpu.zig:106`) has `core` | add `core` to the `qwen27` module and the test module |
| `server/slide_model.zig` | `server/root.zig` in `cuda.zig:394` (`engine_api`, `tokenizer`, `template`, `living`); `server/main.zig` has `core` | add `core` to the `server/root.zig` modules (`cuda.zig:394`, `metal_targets.zig:91`, `cortheon_fuzz.zig:11`) |
| `hip/comm/identity.zig` | `hip/root.zig` | hash-only; no change needed |

## Gates

All on the branch head; base `9e8c368b0`.

- **`zig fmt --check`** on the touched files: clean.
- **`git diff --check`:** clean.
- **Lean:** `python3 -I tools/zig/lean_check.py` gives 424 problems on the base and 424 on the head; `tools/zig/lean_new.py` reports **0 new**.
- **File length:** every touched file is under 600 lines. `pack.zig` is the largest, at 393.
- **One run of 27 steps:**
  - `zig build test`. It includes the core module's tests, where `safetensors_security_test.zig` runs, and `test-checkpoint-security`, which runs `checkpoint_security_test.zig`.
  - Every DeepSeek host step: `test-dsv41-{shared,cuda-host,device-host,state,forward,kv,draft,bounds-host,engram-alloc,assets-host,token-map}`, `test-draft-dsv41-host`, and `compile-dsv41-{cuda,tool,server,runner,device-host,forward,kv,draft,sampler,shared}`.
  - `test-safetensors-header-host`, `test-cluster-checkpoint-host` and `test-cluster-loader-host`.
  - **Result:** 579 of 582 steps succeed; 6,825 of 6,862 tests pass (34 skipped, 3 failed). The 3 failures are all in `test-cuda-host`:
    - `listener`: accepted IPv4 and IPv6 TCP streams disable Nagle;
    - `drama_voice_tool`: cleanup failure warns after publication…;
    - `drama_voice_tool`: cleanup restores private directory permissions….

    The untouched base `9e8c368b0` fails the same three in this sandbox (the BF16-CORE base run, same commit). `test-glm-scheduling-host` and `test-metal-residency-owners-host` were not in this run; no touched file reaches them.
- **The new tests run in these steps and pass.** I ran the built binaries by hand to check:
  - `eaq_container_test` "a shard's header length is read as core reads it…" (core test binary);
  - `pack` "a shard's header length is read as core reads it…" (DeepSeek binary).

## What I could not run

- **No GPU and no macOS.** So the downstream effects of D1, D2, D3 and the `gemma4/xpu_dev` overrun are from reading, not from a run: device-buffer overrun, an unread buffer uploaded, embedding kernels indexing past their buffers, and `load()` panics on large offsets.
- **Not migrated because I can't build them here:** `qwen_vl/weights.zig` and `h3/tensors.zig` (Metal-only).
- **Ports, not the files themselves.** Every copy except `xpu/safetensors.zig` ran as a port, with its Io replaced by reads from memory. Where a copy needs a family or a caller (`slide_model`, `h3` `locate`, `cuda_embed`, `packReady`), the port states what it assumed (see Fuzz setup).
- **`tests/kimi_k3/validate.zig:39-40`** (`data = 8 + n`, read count ignored) is not ported. It runs only after `cluster.checkpoint.load` has checked the same file; it is reachable only if the file changes in between.
- **The real Hugging Face checkpoints** were not used (the host is blocked here). The real layouts in the corpus are synthesized at small sizes with the real names and dtypes.
