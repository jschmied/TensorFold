# ST-FIX report: the defective safetensors readers moved onto core

Follow-up to ST-CORE (`core-safetensors-host`, `ST-REPORT.md`; posted as ashhart/TensorFold#1872).

## Summary

- **Base:** upstream `main` at `9e8c368b0` (`git ls-remote upstream refs/heads/main` when I started).
- **Fixed and pushed:** branch A plus fixes 1-6, one branch each.
- **Stopped:** fix 7 (`xpu/loader.zig` + `xpu/safetensors.zig`), because its build change is larger than a few lines. See "Could not fix".

Every fixed reader:
- crashes on 0 of its fuzz inputs;
- agrees with core on every input, except the rules it keeps on purpose (named per branch);
- has a regression test in the tree, made of the smallest inputs that crashed or fooled it. The test fails on the old code and passes on the new.

| branch | SHA | files | regression test |
|---|---|---|---|
| `core-safetensors-headersize` | `3f4a5fc7e` | `core/eaq_container.zig`, `core/eaq_fixture.zig`, `core/eaq_container_test.zig`, `deepseek_v41/pack.zig` | the two ST-CORE tests (see below) |
| `flashnext-embed-core-header` | `057b41b47` | `flashnext/cuda_embed.zig` | "a malformed embedding entry is refused with an error, and its bytes must match its shape" |
| `glm-dflash-core-header` | `66e34dc1d` | `glm/dflash_load.zig`, `glm/cuda.zig` (one test-list line) | "a malformed drafter header is refused with an error, and a range past the data is refused" |
| `gemma4-xpu-core-header` | `77fa52db3` | `gemma4/xpu_dev.zig` | "a shard header is read as core reads it: wrong sizes, overlaps and malformed entries are refused, names kept" |
| `h3-checkpoint-core-header` | `7d1a4721d` | `h3/checkpoint.zig`, `zig/build/h3_host.zig`, `zig/build/cuda.zig` | "a tensor is found by core's index: not a dtype in another field, not a look-alike in metadata, spaced JSON read" |
| `pull-exlr8-header-check` | `15d80118b` | `cli/pull_exlr8_plan.zig` | "a width file's header run is refetched when its length word or its tensor entries are corrupt" |
| `slide-model-core-header` | `153beae23` | `server/slide_model.zig`, `zig/build/{cuda,metal_targets,cortheon_fuzz}.zig` | "a shard whose header would overflow a sum, a layer or a row read is refused with an error" |

Paths are under `zig/src/` unless they start with `zig/build/`. Each branch is one commit on `9e8c368b0`, authored and committed by Jürgen Schmied, with no trailer.

## How each fix was checked

- **Fuzz re-run.** Each fixed reader was ported into the ST-CORE harness (`/home/user/scratch/st/h/copies3.zig`, not committed) and run against core on the same inputs:
  - main: the 208-file corpus plus 2,500,000 mutants;
  - extra: the 17-file MoE / h3 corpus plus 500,000 mutants.

  "Disagree" counts inputs where the reader's answer is not core's: other tensors, accepting what core refuses, or refusing what core accepts. Every disagreement was traced to the stated rule by replaying the input against core itself:
  - core with `.omit` for the acceptances;
  - core's full parse for the refusals.
- **Old counts** are from the ST-CORE run of the same port of the old code: the same mutant indices, with the main corpus at 206 files (under 5 MB) for the crash-heavy copies.
- **"Before" for the tests.** Each new test's inputs were run against the old code:
  - `cuda_embed`, `pull`: the old function, verbatim, in a scratch test with the new test body;
  - `glm/dflash_load`, `gemma4/xpu_dev`, `h3`, `slide_model`: their API changed, so the old code's ST-CORE port, replayed per input in Debug.
- **Valid files.**
  - On every input the old reader and core both accepted, the old reader gave the same tensors as core (0 "differs" in ST-CORE).
  - The fixed readers agree with core on every input core accepts, except their stated rules.
  - So valid files give the same tensors before and after. The only changes on valid files are inputs that crashed the old code; those are listed per branch.

## Per branch

### Branch A: `core-safetensors-headersize` (`3f4a5fc7e`)

- **What:** the two ST-CORE migrations, unchanged (`eaq_container.readShard` and `deepseek_v41` `Pack.open` read the length with `headerSize`).
- **Base:** upstream main had not moved, so the branch is the two existing commits on `9e8c368b0`, with no report.
- **Before:** own caps of 256 MiB / 1 GiB, allocated before core's 100 MB check. **Now:** core's check first; the same verdict on every input.
- **Fuzz (ST-CORE):** 0 disagreements and 0 crashes, main and extra.
- **Tests:**
  - `eaq_container_test` "a shard's header length is read as core reads it…";
  - `pack` "a shard's header length is read as core reads it…".

  With the eaq length check removed, its test panics (integer overflow).

### 1. `flashnext-embed-core-header` (`057b41b47`): `flashnext/cuda_embed.zig`

- **What changed:**
  - `headerSize`, then `parseHeaderOptions(.unsupported = .omit)`.
  - `tensorAt` reads core's entry, so core's bytes == dtype × shape check now applies. `Table.upload` can no longer take `rows`/`dims` from a shape larger than the buffer.
  - `lookup` uses the same index.
- **Kept on purpose:** a tensor of a dtype or rank core does not name is left out, not refused (`.omit`, as "other tensors, including rank above 4, are ignored" said before). The three tensors must be rank 1 to 4 and of the dtype asked for.
- **Before vs now** (the test's six inputs):

  | input | old | new |
  |---|---|---|
  | `[]` | panic: union field `object` | `BadSafetensors` |
  | `{"…embed_tokens.weight":5}` | panic: union field `object` | `BadSafetensors` |
  | `"data_offsets":[0]` | panic: index out of bounds | `BadSafetensors` |
  | `"data_offsets":[-1,16]` | panic: integer does not fit | `BadSafetensors` |
  | `"data_offsets":[0.0,16]` | panic: union field `integer` | `BadSafetensors` |
  | weight `shape [1000000,4]` over 16 bytes, with valid scales/biases | **accepted** | `BadSafetensors` |

- **Fuzz** (inputs; crashes; disagreements):

  | run | inputs | old crash | new crash | new disagree | of which kept rules |
  |---|---|---|---|---|---|
  | main | 2,500,208 | 237,350 | 0 | 87,012 | 69,010 `.omit` acceptances + 18,002 rank 0/5 refusals |
  | extra | 500,017 | 51,380 | 0 | 16,444 | 13,703 `.omit` + 2,741 rank |

  - Every acceptance replays as accepted by core with `.omit`.
  - The old code accepted 124,785 + 27,889 inputs core refuses; the new code accepts none of them except through `.omit`.
  - The old code also crashed on 8 + 3,229 inputs that core accepts: empty tensors with a dimension past i64, e.g. `"shape":[0,18446744073709551615]`. The new code agrees with core on them.

### 2. `glm-dflash-core-header` (`66e34dc1d`): `glm/dflash_load.zig`

- **What changed:**
  - The file length is read, and both `readPositionalAll` counts are checked (the header length word and the header). The tensor read in `Loader.read` is checked too.
  - `headerSize`, then the new `addEntries`: core's `parseHeader`, then the BF16-only rule.
  - `glm/cuda.zig`'s `test {}` list gains `dflash_load.zig`, so the new test runs in `test-cuda-host`. That is a source line, not build wiring.
- **Kept on purpose:** every tensor BF16 (`DrafterNotBf16`); the 1 MiB header cap (`BadDrafterHeader`).
- **Before vs now** (the test's seven inputs):

  | input | old | new |
  |---|---|---|
  | `[]` | panic: union field `object` | `BadSafetensors` |
  | `{"a":1}` | panic: union field `object` | `BadSafetensors` |
  | no `dtype` | panic: null value | `BadSafetensors` |
  | `"data_offsets":[0]` | panic: index out of bounds | `BadSafetensors` |
  | `"data_offsets":[4,2]` | panic: integer does not fit | `BadSafetensors` |
  | `"data_offsets":[0.0,2]` | panic: union field `integer` | `BadSafetensors` |
  | `"data_offsets":[0,999999]` over 4 data bytes | **accepted** | `BadSafetensors` |

  Also: an unread header or length word is no longer parsed from uninitialised memory.
- **Fuzz:**

  | run | inputs | old crash | new crash | new disagree | of which kept rules |
  |---|---|---|---|---|---|
  | main | 2,500,208 | 67,755 | 0 | 115,786 | 115,783 `DrafterNotBf16` + 3 headers over 1 MiB |
  | extra | 500,017 | 26,784 | 0 | 18,405 | 18,405 `DrafterNotBf16` |

  The new code accepts nothing core refuses. The old code accepted 45,133 + 19,946 such inputs.

### 3. `gemma4-xpu-core-header` (`77fa52db3`): `gemma4/xpu_dev.zig`

- **What changed:**
  - `parseShard` is `headerSize` + `parseHeader`. It returns the arena that holds the names, and `Checkpoint.headers` keeps those arenas.
  - `Tensor` keeps its string dtype: core's tag in upper case, which is the safetensors name. So `xpu_load`'s `"BF16"` comparisons are unchanged.
  - With core's size check, `projection8` / `projection` can no longer copy more bytes than the shape they sized the device buffer from.
- **Kept on purpose:** none. It is plain core.
- **Before vs now** (the test's five malformed inputs):

  | input | old | new |
  |---|---|---|
  | `[]` | refused | refused |
  | `{"w":{"dtype":"U8","shape":[],"data_offsets":[0,4]}}` (4 bytes for 1) | **accepted** | refused |
  | two tensors on `[0,2]` and `[1,3]` | **accepted** | refused |
  | dtype `Q4` | **accepted** | refused |
  | rank 6 | **accepted** | refused |

  And a valid shard is read with the right dtype, rank and bytes.
- **Fuzz:** main 2,500,208 and extra 500,017 inputs. 0 disagreements and 0 crashes. The old code had 221,599 + 40,987 looser and 17 stricter.
- **Running the test:**
  - No host step runs `xpu_dev.zig`'s tests: its module is only tested under the `draft sources:` / `slide admission:` filters.
  - I ran the new test with `xpu_dev.zig` as the test root, with the `ze` and `core` modules the build gives it. The command is taken from `zig build test-draft-xpu-host --verbose`. Result: 1/1 OK.
  - `test-draft-xpu-host` compiles the module.

### 4. `h3-checkpoint-core-header` (`7d1a4721d`): `h3/checkpoint.zig`

- **What changed:**
  - `Checkpoint.openWith` checks the length with `headerSize`, then indexes the header with `parseHeader` into an arena the checkpoint owns (freed in `deinit`).
  - `locate` reads core's entry; there is no text search.
  - `header` is kept for `cuda_weights.blockCount`, which already parses JSON.
- **Build change:** the h3 CUDA modules imported only `cuda`, and `checkpoint.zig` cannot reach core by a relative path. So I added `pub fn safetensors(b, target)` to `zig/build/h3_host.zig`: a module rooted at `core/safetensors.zig`, the same module the Metal h3 tree uses. It is added to:
  - `h3_host.zig`'s `module()` and the `tf-h3` library;
  - `cuda.zig`'s `h3_checkpoint` test module.

  That is 9 lines in two build files. None of these trees holds another core, so the file is in one module per build.
- **Kept on purpose:** BF16 only (`WrongDType`); rank 1 or 2 with nonzero dimensions (`BadHeader`); files under 16 bytes (`CannotOpen`).
- **Old tests:** their literal headers now go through core: the size-mismatch and overflow cases are refused at the index, and a rank-3 BF16 tensor stands in for the "BadHeader from locate" case.
- **Before vs now:**

  | input | old | new |
  |---|---|---|
  | `{"x":{"dtype":"F16",…,"n":"BF16"}}` | `x` read as BF16 | `WrongDType` |
  | a BF16 look-alike inside `__metadata__`, then `a` at [2,4] | `a` at [0,2] | `a` at [2,4] |
  | spaced JSON `{"a": {"dtype": "BF16", …}}` | `MissingTensor` | found at [0,2] |
  | trailing `x` after the JSON | **accepted** | refused |
  | duplicate key | **accepted** (first wins) | refused |

- **Fuzz:**

  | run | inputs | new crash | new disagree | of which kept rules |
  |---|---|---|---|---|
  | main | 2,500,208 | 0 | 7,552 | 7,518 rank/zero-dimension `BadHeader` + 34 files under 16 B |
  | extra | 500,017 | 0 | 1,344 | 1,336 + 8 |

  - All are refusals; the new code accepts nothing core refuses and gives no other tensors ("differs" 0).
  - The old code had 1,297,944 + 229,039 looser, 121,997 + 17,494 stricter and 6,329 differing.

### 5. `pull-exlr8-header-check` (`15d80118b`): `cli/pull_exlr8_plan.zig`

- **What changed:**
  - `headerOk` compares the length word with `header_bytes - 8`, so there is no `n + 8` overflow.
  - It requires `t.size >= header_bytes`.
  - It runs `exlr8.safetensors.parseHeader` on the run over `t.size - header_bytes` data bytes, so a corrupt run fails and is refetched.
  - The existing test now gives its target the real size (`header_bytes`), not 0.
- **Kept on purpose:** the `__metadata__` rule (format, contents, layer, k).
- **Before vs now:**
  - Length words 2^64−1 and 2^64−8: **panic (integer overflow)** in Debug/ReleaseSafe → false.
  - Entries `"t":0`, a range past the data, or dtype `Q4`: **true** (kept) → false (refetched).
- **Fuzz** (a dedicated rerun, since `headerOk` sees only 65,536-byte runs): 1,000,000 mutants of a width header with tensor entries, padded to 65,528 bytes, with one in eight given a random length word. 71,016 accepted; 0 disagreements with core + the metadata rule; 0 crashes (ReleaseSafe).

### 6. `slide-model-core-header` (`153beae23`): `server/slide_model.zig`

- **What changed:**
  - `readShard`: `headerSize` against the shard's size, then `parseHeaderOptions(.omit)`.
  - Checked arithmetic: `std.math.add` for `expert_bytes`, and `std.math.cast` to u16 for the layer (error, not panic).
  - `inUse` takes core's entry. It needs rank 2, computes `row` and `rows × row` with checked multiplies, and reads only if those rows fit inside the tensor's range, which core checked against the file. So its allocation is bounded by the file.
  - `offsets` is gone.
- **Build change:** `core` is added to the `server/root.zig` test modules: `cuda.zig` (`mods.core`) and `metal_targets.zig` (`engine`, whose `tensorfold.zig` exports `safetensors`). The cortheon fuzz harness gets a `core` parameter in `cortheon_fuzz.add` and its two callers. That is 6 lines in 3 build files. The `server/main.zig` modules already had `core`.
- **Kept on purpose:** entries of an unnamed dtype or rank > 5 are left out (`.omit`; the old code skipped what it could not read). A layer above 65535 refuses the shard.
- **Before vs now** (the test's four inputs, pocket 1):

  | input | old | new |
  |---|---|---|
  | three shared-expert entries with `[0,2^63−1]` | panic: integer overflow (84) | `BadSafetensors` |
  | layer 65536 | panic: integer does not fit (85) | `BadSafetensors` |
  | up scales `[2^62,2]` | panic: integer overflow (121) | `BadSafetensors` |
  | up scales `[2,2^35]` | 64 GiB allocation | `BadSafetensors` |

  The existing SwiGLU pocket test still passes.
- **Fuzz** (this reader compares its length step with `headerSize`, so each refusal was replayed against core's full parse):

  | run | inputs | new crash | refused where core's full parse accepts |
  |---|---|---|---|
  | main | 2,500,208 | 0 | 0 |
  | extra | 500,017 | 0 | 2,390, all with a shared-expert layer above 65535 (the kept rule) |

  The old code crashed on 9,804 of the extra inputs.

## Could not fix

- **Fix 7 (`xpu-loader-core-header`), stopped as the brief allows.** `xpu/loader.zig` and `xpu/safetensors.zig` would need `core`. Their modules are built in several places that already hold *separate* `core` module instances:
  - `xpu/root.zig` in `runtime()` (`zig/build/xpu.zig:188`, called at `:215` and `:279`);
  - `qwen27` at `:133` and `:284`;
  - `xpu/safetensors.zig` as its own test root in `build.zig:123`;
  - `core` created at `xpu.zig:97` and `:299`, and inside `family()`.

  A tree that holds two instances fails with "file exists in multiple modules". So the change means passing one `core` instance through `runtime()`, `family()` and the module builders in `xpu.zig`, plus `build.zig`, rather than a few one-line imports.

  Either option keeps D1's defects open until it is done: the allocation from the length word, 12 panic kinds, and offsets not checked against the file. The alternative of a standalone `safetensors` module, as for h3, needs the same check that no XPU tree also holds `core`, and `qwen27_main` (`:106`) does.
- **Not run here:**
  - The Metal build (macOS only), so `metal_targets.zig`'s one-line change in fix 6 was not compiled.
  - The XPU device build (`xpu-gemma4` needs ocloc and nGEN), so fix 3 was compiled and tested as a host module only.
  - No GPU.

## Gates

On each branch head. In every row:
- **Clean:** `zig fmt --check` on the touched files, `git diff --check`, and `lean_check.py` + `lean_new.py` (424 on the base, 424 on the head, **0 new**).
- **Under 600 lines:** every touched file.
- **Known failures:** the 3 sandbox failures that the untouched base `9e8c368b0` also has, all in `test-cuda-host`:
  - `listener`: accepted IPv4 and IPv6 TCP streams disable Nagle;
  - `drama_voice_tool`: cleanup failure warns after publication…;
  - `drama_voice_tool`: cleanup restores private directory permissions….

| branch | steps run | result |
|---|---|---|
| A | `zig build test` + all DeepSeek host and compile steps + `test-checkpoint-security`, the safetensors and cluster header steps (the ST-CORE run, same tree) | 579/582 steps; the 3 known |
| 1 | `zig build test` | 455/458; the 3 known; new test 43/171 OK |
| 2 | `zig build test`, `test-glm-dflash-host`, `test-glm-contract-host`; then `test-cuda-host` again with the `cuda.zig` line | 458/461 and 279/281; the 3 known; new test 51/159 OK |
| 3 | `zig build test`, `test-draft-xpu-host`, `test-slide-xpu-host`; new test as above | 456/462; the 3 known, plus `test-slide-xpu-host`'s compile error in `gemma4/xpu_forward.zig:147` (`draft_xpu_images` has no `rows`). The base fails the same way; branch `fix-slide-xpu-host-table` fixes it. New test 1/1 OK |
| 4 | `zig build test`, `test-h3-{geometry,ownership,metadata,arithmetic,blocks,boundaries}`, `compile-h3-library` | 477/480; the 3 known; the h3 checkpoint binary 6/6 OK |
| 5 | `zig build test` | 455/458; the 3 known; new test 30/30 OK |
| 6 | `zig build test`, `test-server-recovery-host`, `test-server-cpu`, `test-server-security`, `fuzz-cortheon` | 466/470; the 3 known (in `test-server-cpu` and `test-cuda-host`, the same binary); new test 88/923 OK |

**A sandbox note:** one `test-cuda-host` run on branch 4 also failed `recall_routes` and `late_system_test` when free disk fell to 3.4 GB. After freeing space (compressing my fuzz logs) both passed, and the run was back to the 3 known failures.
