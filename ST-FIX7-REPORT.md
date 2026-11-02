# ST-FIX7 report: the XPU safetensors readers moved onto core

Follow-up to ST-FIX (`ST-FIX-REPORT.md`, fix 7 stopped there) for ST-CORE (ashhart/TensorFold#1872).

## Summary

- **Base:** upstream `main` at `3ce4a0480` (fetched when I started).
- **Done and pushed:** branch 1 (one `core` through the XPU build) and branch 2 (the two readers on core, on branch 1).
- **Branch 3 not needed:** the fix on `fix-slide-xpu-host-table` is already in main. Nothing was pushed for it (see Branch 3).

| branch | SHA | files | build lines |
|---|---|---|---|
| `xpu-build-one-core` | `1cfaed755` | `zig/build/xpu.zig`, `build.zig` | 25 added, 21 removed (46) |
| `xpu-loader-core-header` | `971d74781` | `xpu/loader.zig`, `xpu/safetensors.zig`, `xpu/dev.zig` (1 line), `xpu/root.zig` (1 test-list line) | none |

Paths are under `zig/src/` unless they start with `zig/build/` or are `build.zig`. Each branch is one commit, authored and committed by Jürgen Schmied, with no trailer.

## Branch 1: `xpu-build-one-core` (`1cfaed755`)

### What changed

- **`steps()` (the `-Dxpu` and `xpu-*` device trees):**
  - The existing ReleaseSafe `core` gets the `gpu_memory` import that `family()`'s core had.
  - That one `core` goes to `runtime()` (the `xpu` module) and `family()` (Nemotron) through `nemotronModules()`, and to the `qwen27` module of `xpu-native`.
  - So the `xpu-native` tree (server, CLI, Gemma 4, Qwen27, Nemotron, the XPU runtime) now holds one `core`. On the base it held two: the ReleaseSafe one and `family()`'s.
  - The `xpu-qwen27` runner keeps its own `core` (its tree's only one, at the `optimize` mode as before). It is now named `qwen_core`, and the three `xpu-qwen27-tests` programs import it.
- **`runtime()`** takes `core` and adds it to the `xpu` module. **`family()`** takes `core` instead of creating one.
- **`hostTests()`:**
  - One host `core` is made first and used by `runtime()`, `family()`, the `qwen27` module, `gemma_native` (as before, through `mods.core`) and the `xpu_config` test (it had its own `core` before).
  - New in the same test loop: `xpu/safetensors.zig` as a test root with that `core`, and `gemma4/xpu_dev.zig` as a test root with `ze` and that `core`.
- **`build.zig`:** the `xpu/safetensors.zig` test line is removed; the test now runs in `hostTests()` (above).

### One behaviour change to name

The Nemotron device trees (`tensorfold-xpu`, `tf-xpu-test`, `xpu-tests`) now get the shared ReleaseSafe `core`; before, `family()` built theirs at the `optimize` mode (default fast). The code is the same; only `core`'s own functions are built with safety checks in those binaries.

The other option was to build the Nemotron modules twice (once per mode), which doubles their 14 SPIR-V steps and adds build lines. I chose the smaller change.

### Proof of no behaviour change

- **`zig build --help`:** the same 526 lines on base and branch, also with `-Dxpu` (same md5).
- **XPU steps, base `3ce4a0480` vs the branch heads:** the same results.

  | step | how | base | branch 1 | branch 2 |
  |---|---|---|---|---|
  | `test-slide-xpu-host` | host | OK, 2/2 tests | OK, 2/2 tests | OK, 2/2 tests |
  | `test-draft-xpu-host` | host | OK, 2/2 tests | OK, 2/2 tests | OK, 2/2 tests |
  | `-Dxpu xpu-spirv` | real ocloc, `-Dxpu-device=tgllp` | 29/29 steps | 29/29 | 29/29 |
  | `-Dxpu xpu-tests` | real ocloc, `tgllp` | 44/44 | 44/44 | 44/44 |
  | `xpu-gemma4` | fake ocloc + nGEN stub (below) | 14/14 | 14/14 | 14/14 |
  | `xpu-qwen27` | fake ocloc + nGEN stub | 14/14 | 14/14 | 14/14 |
  | `xpu-qwen27-tests` | fake ocloc + nGEN stub | 20/20, 11/11 tests | 20/20, 11/11 tests | 20/20, 13/13 tests (the two new `xpu/safetensors.zig` tests, collected through `qwen27.zig`) |
  | `xpu-native` | fake ocloc + nGEN stub | 31/31 | 31/31 | 31/31 |
  | `-Dxpu xpu-tests` | fake ocloc + nGEN stub | 44/44 | 44/44 | 44/44 |
  | `zig build test` (runs `hostTests`) | host | see Gates | see Gates | see Gates |

- **About the tools here:**
  - The ocloc on this host knows `tgllp` but not `bmg` (the default device). So the Qwen27 and Gemma 4 kernels, which need DPAS / Xe2, do not compile, and there are no nGEN headers.
  - For the device trees I compiled the Zig code with a scratch-only fake ocloc (it writes empty `.bin` / `.spv` files) and a local, uncommitted patch that replaces the nGEN generator with empty `.zebin` files. The same fake and patch were used on base and branch.
  - This shows that the Zig trees compile and link. It does not run any kernel.

### The `gemma4/xpu_dev.zig` host tests

- They now run in `zig build test`, through `hostTests()`.
- On main, `xpu_dev.zig` has no tests yet: they come with fix 3 (`gemma4-xpu-core-header`). So this step runs 0 tests until fix 3 is merged. Its module already has the `core` import that fix 3 needs.
- I added them to an existing step rather than a new named step, so `zig build --help` stays the same.

## Branch 2: `xpu-loader-core-header` (`971d74781`, on branch 1)

### `xpu/loader.zig` (D1, the Nemotron-H XPU loader)

- **What changed:**
  - `addShard` calls the new `shardHeader`:
    - the file size from `lseek`;
    - `headerSize` on the length word, before any allocation;
    - the header read into an arena;
    - `parseHeaderOptions(.{ .unsupported = .omit })` with the data length.
  - Each entry becomes an `Info` at `8 + header + begin`, with `len = end - begin`. Its dtype is core's tag in upper case (the safetensors name), so `loadF32`'s `"BF16"` check is unchanged.
  - The fd is closed when the header is refused. Before, it leaked on every error.
  - Offsets are now inside the file, so `load()`'s `@intCast(off + done)` and `loadF32`'s allocation are bounded by the file size.
- **Kept on purpose: `.omit`.** A dtype or a rank core does not name (rank > 5) is left out of the index, not refused. The old loader accepted any dtype string and any rank, so a shard with such a tensor still loads. Asking for that tensor now gives `MissingTensor` instead of its bytes.

### `xpu/safetensors.zig` (the Qwen27 runner's reader, through `xpu/dev.zig`)

- **What changed:**
  - `parse` is `headerSize` + `parseHeader` into an arena. It returns the arena that holds the names, and `Checkpoint.headers` keeps those arenas (the one `dev.zig` line).
  - `Tensor` is unchanged: string dtype (core's tag in upper case), `shape: [8]u64` padded with ones, `rank`, `bytes`.
  - The tensors keep the header's order (core's index keeps insertion order), so `Checkpoint.get`'s first match is unchanged.
- **Kept on purpose:** its error names. Core's `UnsupportedDType` → `UnknownDtype`, `RankTooHigh` → `TensorRank`, every other refusal → `BadSafetensors`.

### What the readers refused before and refuse now

| input | `loader` old | `loader` new | `safetensors` old | `safetensors` new |
|---|---|---|---|---|
| malformed JSON fields (the 14 test inputs below) | **panic** | refused | refused | refused |
| length word `0xffff…ffff` | **allocates 2^64−1** (OutOfMemory) | refused | refused | refused |
| length word past the file | **allocates it**, then `ReadFailed` | refused | refused | refused |
| header over core's 100 MB cap | read | refused | read | refused |
| offsets past the file | **accepted** | refused | refused | refused |
| bytes ≠ dtype × shape | **accepted** | refused | refused | refused |
| overlapping tensors | **accepted** | refused | **accepted** | refused |
| rank 6–8 | **accepted** | left out (`.omit`) | **accepted** | refused (`TensorRank`) |
| rank > 8 | **accepted** | left out | refused | refused |
| dtype core does not name (`Q4`, `F4`, `C64`, lowercase) | **accepted** | left out | refused | refused |
| `F8_E8M0` | accepted | accepted | **refused** | accepted |

### Fuzz re-run

Both fixed readers were ported into the ST-CORE harness (`copies3.zig`, not committed):
- `fixed.xpu_safetensors` is the new file itself, with its `core` import pointed at the harness's copy of core;
- `fixed.xpu_loader` is `shardHeader` + `addShard`, with the fd reads as reads of the bytes in memory.

| | inputs | crashes | disagree with core | over 64 MiB and 4× the file |
|---|---|---|---|---|
| `fixed.xpu_loader` main | 2,500,208 | **0** | 73,736 looser | 0 |
| `fixed.xpu_loader` extra | 500,017 | **0** | 13,716 looser | 0 |
| `fixed.xpu_safetensors` main | 2,500,208 | **0** | 0 | 1 (below) |
| `fixed.xpu_safetensors` extra | 500,017 | **0** | 0 | 0 |
| old `xpu_loader.addShard` main / extra (ST-CORE) | 2,500,206 / 500,017 | 327,874 / 64,888 | 405,755 / 76,504 looser | 66,635 / 13,088 |
| old `xpu_safetensors.parse` main / extra (ST-CORE) | 2,500,208 / 500,017 | 0 / 0 | 27 / 2 looser, 8,721 / 109 stricter | 0 / 0 |

- **Every loader disagreement is the `.omit` rule.** I regenerated all 73,736 + 13,716 inputs by index and ran core's `parseHeaderOptions(.omit)` on them: core accepts every one (0 other).
  - Examples: `{"w": {"dtype": "C64", "shape": [1], "data_offsets": [0, 8]}}`, a dtype `"F4"`, a dtype `"u8"`.
- **The one large request:** `json_large_60k_entries.st`, a valid corpus file with a 4.4 MB header and 60,000 tensors. The new `parse` asks its arena's backing allocator for 84.6 MB in one block.
  - The reason: core's JSON tree and its index share one arena. The old code kept only the JSON tree, which stayed under 64 MiB.
  - The amount follows the header size, and core caps the header at 100 MB. Core's other callers (`File.open`) work the same way.

### Valid files: the same tensors before and after

- On every corpus file core accepts (both corpora), I compared old and new on name, order, dtype, shape and byte range.
  - `xpu/safetensors.zig`: 84 files the same; 5 files read only by the new code (`F8_E8M0`, which the old code refused).
  - `xpu/loader.zig`: 89 files the same.
  - 0 differ.
- On the corpus files core refuses, the new loader accepts 16. All 16 are the `.omit` rule: rank 6, 8, 9 or 40; dtype `F4`, `C64`, lowercase or unknown; and fixtures of the same kinds.

### Regression tests

- **`xpu/loader.zig`:** "a shard header is read as core reads it: malformed entries, a huge length word and offsets past the file are refused". It writes each input to a temp file and calls `addShard`. It also checks that a valid shard is read with the right offset, length and dtype, and that a `Q4` tensor is left out.

  The inputs, and each one replayed on the old code (the ST-CORE port, Debug) and on the new:

  | header (after the length word) | old | new |
  |---|---|---|
  | `"data_offsets":[]` | panic: index out of bounds: index 0, len 0 | `BadSafetensors` |
  | `[0]` | panic: index out of bounds: index 1, len 1 | `BadSafetensors` |
  | `[-1,2]` | panic: integer does not fit in destination type | `BadSafetensors` |
  | `[-0,2]` | panic: access of union field 'integer' while field 'float' is active | `BadSafetensors` |
  | `[true,2]` | panic: … while field 'bool' is active | `BadSafetensors` |
  | `[null,2]` | panic: … while field 'null' is active | `BadSafetensors` |
  | `"4"` | panic: access of union field 'array' while field 'string' is active | `BadSafetensors` |
  | `[18446744073709551616,2]` | panic: … 'integer' while field 'number_string' is active | `BadSafetensors` |
  | `[7,2]` | panic: integer overflow | `BadSafetensors` |
  | `[]` (the root) | panic: access of union field 'object' while field 'array' is active | `BadSafetensors` |
  | `{"a":0}` | panic: … 'object' while field 'integer' is active | `BadSafetensors` |
  | no `data_offsets` | panic: attempt to use null value | `BadSafetensors` |
  | no `dtype` | panic: attempt to use null value | `BadSafetensors` |
  | `"dtype":1` | panic: access of union field 'string' while field 'integer' is active | `BadSafetensors` |
  | `[0,4]` with 2 data bytes | **accepted** | `BadSafetensors` |
  | length word `0xffffffffffffffff`, header `{}` | requests 2^64−1 bytes: OutOfMemory from the real allocator; in the test's arena, panic: integer overflow | `BadSafetensors` |
  | length word 1 MiB, a 10-byte file | allocates 1 MiB, then `ReadFailed` | `BadSafetensors` |

  The first nine use `{"a":{"dtype":"U8","shape":[0],"data_offsets":…}}`.

  **Before:** the test file with the old `addShard` panics on its first input:
  `panic: index out of bounds: index 0, len 0` at `loader.zig:99` (`const lo: u64 = @intCast(offs[0].integer);`), test command terminated with signal ABRT.
  **Now:** 1/1 OK.
- **`xpu/safetensors.zig`:** two new tests next to its four old ones.
  - "a rank past core's, overlapping tensors or a header past core's cap are refused; F8_E8M0 is read". **Before** (the old `parse` with the new tests): `expected error.TensorRank, found 1`, 5 passed, 1 failed. **Now:** OK.
  - "tensors keep the header's order, names, dtype spelling, shape padded with ones and bytes". It passes on both: it guards the unchanged behaviour.
  - All 6: OK.
- **How I ran them:** with `zig test` on each file as the test root, with the modules the build gives it (`core`, and for the loader also `ze`, `sys` and a `kernel_options` file). In the tree, they run in `zig build test`: the loader test through `xpu/root.zig`'s test list (the new `_ = loader;` line), the safetensors tests through branch 1's test root.

## Branch 3: `fix-slide-xpu-host-table` (not needed)

- Main already has this fix: `3f2d0ed64` "build: the XPU host tests' empty kernel table names every image the engines read". It has the same patch-id as the branch's `5d26ca070`.
- Rebasing the branch onto the base would leave an empty commit, so I did not rebase or push it.
- `test-slide-xpu-host` compiles and passes on the base and on both branches (2/2 tests).

## Gates

On each branch head:
- **Clean:** `zig fmt --check` on the touched files, `git diff --check`, and `lean_check.py` + `lean_new.py` (424 on the base, 424 on each head, **0 new**). A first branch 1 commit had one build comment at 124 columns; I shortened it before pushing.
- **Under 600 lines:** every touched file (`zig/build/xpu.zig` is the largest, at 306).
- **Commits:** author and committer Jürgen Schmied, no trailer.

| branch | steps run | result |
|---|---|---|
| 1 | `zig build test test-cuda-host`; the 9 XPU steps above | 464/467 steps; 6,154/6,203 tests (46 skipped, 3 failed): the 3 known; XPU steps as on base |
| 2 | the same, in the same tree after branch 1 (so unchanged steps were reused) | 464/467 steps; 1,134/1,140 tests run (3 skipped, 3 failed): the 3 known; XPU steps as on base. The built test binaries run by hand: the `xpu` module 5/5 (the loader test among them), `xpu/safetensors.zig` 6/6 |

**The 3 known failures**, all in `test-cuda-host`:
- `listener`: accepted IPv4 and IPv6 TCP streams disable Nagle for incremental responses;
- `drama_voice_tool`: cleanup restores private directory permissions after encoder exit;
- `drama_voice_tool`: cleanup failure warns after publication and accompanies encoder failure.

The untouched base `3ce4a0480` fails the same three: `zig build test-cuda-host` there gives 283/285 steps, 4,310/4,330 tests (17 skipped, 3 failed), the same 3 names.

**XPU steps not run for real:** see "What I could not do".

## What I could not do

- **Run XPU kernels or devices.** There is no Intel GPU here, the local ocloc knows `tgllp` but not `bmg`, and there are no nGEN headers.
  - Not run for real: `xpu-gemma4`, `xpu-qwen27`, `xpu-qwen27-tests`, `xpu-native`, and `-Dxpu` with the default device.
  - They were compiled with a fake ocloc and an nGEN stub (Branch 1), on base and branch alike.
  - `-Dxpu xpu-spirv` and `xpu-tests` were built with the real ocloc for `tgllp`.
- **`gemma4/xpu_dev.zig`'s tests run 0 tests on main.** Its tests come with fix 3; once fix 3 is merged, they run in `zig build test` through branch 1.
- **Ports, not the files.** The loader ran as a port in the fuzz harness: its reads from the fd became reads from memory. `xpu/safetensors.zig` ran as the file itself.
