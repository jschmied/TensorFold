# XPU-REVIEW report: the maintainer's ask on ashhart/TensorFold#1918 (and #1919)

ashhart's comment on #1918 asked for two things:
1. the Nemotron XPU device binaries keep a `core` at `optimize`, even if the XPU tree builds `core` twice;
2. after #1905 lands, both PRs are rebased onto it, with #1918's own `xpu_dev.zig` test entry dropped.

Both steps ran: #1905 is in main (its changelog is `d114693d1`; the build part is `2aefc3772`).

## SHAs

| | before | after | push |
|---|---|---|---|
| Step A, `xpu-build-one-core` (on main `3ce4a0480`) | `1cfaed755` | `c59fb1497` (one new commit) | fast-forward |
| Step B, `xpu-build-one-core` (on main `9bb51cd63`) | `c59fb1497` | `89657d210` (`42b2607ca` + `89657d210`) | `--force-with-lease` |
| Step B, `xpu-loader-core-header` (on the new `xpu-build-one-core`) | `971d74781` | `63250bf48` | `--force-with-lease` |

All commits are authored and committed by Jürgen Schmied, with no trailer. No PR comments and no new PRs.

## Step A: the Nemotron device binaries keep `optimize` (`c59fb1497`)

**What changed** (`zig/build/xpu.zig`, 9 lines added, 5 removed):
- The `core` that the `xpu-qwen27` runner already built at `optimize` is renamed `runner_core`. It gets the `gpu_memory` import that `family()`'s core had on main.
- The Nemotron device binaries (`tensorfold-xpu`, `tf-xpu-test`, the 14 `tf-xpu-*-test` programs) now take a second Nemotron module set, built over `runner_core`:
  - `family(…, runtime(…, &nem.images, ze, runner_core, dpas), runner_core, lanes)`;
  - the SPIR-V images are shared, so no ocloc step is added.
- The `xpu-native` tree (server, CLI, Gemma 4, Qwen27, Nemotron, the XPU runtime) keeps its one ReleaseSafe `core`.
- So the Nemotron and `xpu` runtime modules are built once per mode, and no tree holds two `core`s.

**Why `runner_core` and not a third `core`:** the `xpu-qwen27` runner and the Nemotron device binaries are separate trees that both want `core` at `optimize`. Sharing one module object across two trees changes nothing in either compile.

## Step B: rebased onto main `9bb51cd63`

- **`xpu-build-one-core`:** both commits rebased.
  - The one conflict was `hostTests()`. Main (#1905) added `dev` / `dev_tests` for `gemma4/xpu_dev.zig`, feeding `test-draft-xpu-host` and `step`. I kept main's entry and dropped #1918's own `xpu_dev.zig` test root; only `xpu/safetensors.zig`'s test root remains in the loop.
  - The first commit's message no longer mentions `xpu_dev.zig`.
- **`xpu-loader-core-header`:** rebased onto the new `xpu-build-one-core` with no conflicts. Its tree is identical to the pre-reword rebase.
- **`xpu_dev.zig` test compiles** in the whole XPU build graph: main 1, #1918 as posted 1 (main had no such entry then), Step A 1, Step B 1 and 1. So its tests run once.

## Module graph per tree, before and after

**How:** the build runner was run with a logging stand-in for `zig` that records each compile's arguments (`-O` and `-M` per module) and fails the compile.
- Steps: `-Dxpu install xpu-tests xpu-spirv xpu-native xpu-gemma4 xpu-qwen27 xpu-qwen27-tests test-slide-xpu-host test-draft-xpu-host test`, with the stub ocloc and nGEN.
- Each line below lists the modules whose root is `zig/src/core/root.zig`, with their mode: `fast` is ReleaseFast (the default `optimize`), `safe` is ReleaseSafe.

| tree (compile) | main `3ce4a0480` / `9bb51cd63` | #1918 as posted `1cfaed755` | Step A `c59fb1497` / Step B `89657d210`, `63250bf48` |
|---|---|---|---|
| `xpu-native`: `tensorfold-native` | **two:** `core@safe`, `core0@fast` | `core@safe` | `core@safe` |
| `xpu-gemma4` | `core@safe` | `core@safe` | `core@safe` |
| `xpu-qwen27` | `core@fast` | `core@fast` | `core@fast` |
| `xpu-qwen27-tests`: `close` / `means` / `seam` programs | no core | `core@fast` | `core@fast` |
| `xpu-qwen27-tests`: the `qwen27.zig` host run | no core | `core@safe` | `core@safe` |
| Nemotron: `tensorfold-xpu`, `tf-xpu-test`, 14 × `tf-xpu-*-test` | `core@fast` | **`core@safe`** | `core@fast` |
| host tests: `xpu/root.zig`, `xpu/safetensors.zig` | no core | `core@debug` | `core@debug` |
| host tests: `nemotron/xpu.zig`, `cli/xpu_main.zig`, `draft_sources_xpu_test.zig`, `xpu_config.zig`, `xpu_dev.zig` | `core@debug` | `core@debug` | `core@debug` |

- **No compile holds two `core`s after the change.** On main, `tensorfold-native` held two.
- **The Nemotron device binaries match main exactly**, comparing every module's path and mode, not only `core`: same modules, same modes for `tensorfold-xpu`, `tf-xpu-test` and all 14 `tf-xpu-*-test` (Step A against `3ce4a0480`, Step B against `9bb51cd63`).
- **The only differences from main are where a `core` import was added for the readers:**
  - the three `xpu-qwen27-tests` programs gain `core@fast` (and `gpu_memory`);
  - the `qwen27.zig` host run and the `xpu` / `safetensors` host tests gain a `core`.
- **Branch 2 adds no module:** `63250bf48`'s graph is identical to `89657d210`'s.

## Gates

On each pushed head:
- **Clean:** `zig fmt --check` on the touched files and `git diff --check`.
- **Lean, 0 new:**
  - Step A: 424 on `3ce4a0480`, 424 on `c59fb1497`;
  - Step B: 424 on `9bb51cd63`, 424 on `89657d210` and on `63250bf48`.
- **Under 600 lines:** every touched file (`zig/build/xpu.zig` is 315 after Step B).
- **`zig build --help`:** the same steps as main, with and without `-Dxpu` (Step A against `3ce4a0480`).

| head | `zig build test test-cuda-host` | the 9 XPU steps (as for ST-FIX7) |
|---|---|---|
| Step A `c59fb1497` | 464/467 steps; 6,154/6,203 tests; 46 skipped; the 3 known failures | all as on main (table below) |
| Step B `89657d210` | 459/470 steps; 5,073/5,095 tests; 19 skipped; the 3 known failures, plus main's qwen4 compile error (3 steps, below) | all as on main |
| Step B `63250bf48` | 459/470 steps; the 3 known failures and the same qwen4 compile error; run by hand: the `xpu` module 5/5 (the loader test among them), `xpu/safetensors.zig` 6/6, `xpu_dev.zig` 1/1 | all as on main; `xpu-qwen27-tests` 13/13 tests |

The 3 known sandbox failures, all in `test-cuda-host`, are the same on main:
- `listener`: accepted IPv4 and IPv6 TCP streams disable Nagle;
- the two `drama_voice_tool` cleanup tests.

**Main's own failure on `9bb51cd63`:** three qwen4 host compiles (`test-qwen4-host`, `test-qwen4-native-mixers-host`, `test-qwen4-device-host`) fail with `qwen4_cuda_attention_host.cpp:17:5: error: use of undeclared identifier 'fn_attn_merge_body'`. Untouched main `9bb51cd63` fails the same 3 steps here (11/20 steps, the same error), and neither branch touches qwen4. It is not from these PRs.

**The 9 XPU steps:**
- **Ran for real here:**
  - `test-slide-xpu-host` and `test-draft-xpu-host` (2/2 tests each);
  - `-Dxpu xpu-spirv` (29/29) and `-Dxpu xpu-tests` (44/44), with the real ocloc for `-Dxpu-device=tgllp`.
- **Compiled with a stub ocloc and nGEN, so the Zig compiles and links but no kernel runs:**
  - `xpu-gemma4` (14/14);
  - `xpu-qwen27` (14/14);
  - `xpu-qwen27-tests` (20/20 steps, 11/11 tests; 13/13 on the loader head, which adds `xpu/safetensors.zig`'s two new tests);
  - `xpu-native` (31/31);
  - `-Dxpu xpu-tests` (44/44).

**Not run:** the default `bmg` device and real nGEN (the local ocloc knows `tgllp` only, and there are no nGEN headers), and anything on an Intel GPU.

## Step B: ran

Yes. #1905 was in main when Step A was done, so both branches were rebased and pushed with `--force-with-lease`.
