# ROPE-CORE: one RoPE inverse-frequency helper in core

CPU-only work. Delete this file before any PR.

- **Branch:** `core-rope-freq` from `ashhart/TensorFold` main at `ed2858298`. Commits: the helper, one family migration, this report.

## Tools

- **Zig:** 0.17.0 from the PyPI wheel (ziglang.org is blocked here).
- **Python:** `python3 -I`.
- **References:** torch 2.15.0 (PyPI, run on CPU) and transformers 5.22.1.
  - The CPU is an Intel Xeon; torch reports AVX-512.
  - `ATEN_CPU_CAPABILITY=default` and `avx2` select its other code paths.
- **Not downloaded:** Hugging Face (`huggingface.co`) is blocked by this environment's network policy (403). No checkpoint `config.json` came from Hugging Face. The configs below come from:
  - the repo's fixtures (`families/*/fixtures/*.json`);
  - the families' own pinned constants;
  - transformers' own rope code (read from the 5.22.1 wheel).
- **Method:** each copy's expression is ported verbatim (same types, casts and order) into a small Zig program (scratch, not committed). It prints the fp32 words for every config the copy serves. A Python script computes transformers' words with CPU torch and compares.

## What transformers does, and the helper's form

- **The expression:** every rope class the audited families use computes `inv_freq = 1.0 / (base ** (torch.arange(0, d, 2, dtype=torch.float) / d))`, in fp32.
  - Default rope: `modeling_rope_utils` and the per-model `compute_default_rope_parameters`.
  - Proportional rope: `_compute_proportional_rope_parameters`, over `head_dim`, zeros past `rope_angles`.
  - Vision rotaries: Gemma 4, Qwen2-VL, Qwen3-VL, GLM-4V and MiniMax-M3, with `spatial_dim = head_dim // 2`, or `2·((head_dim // 3) // 2)` for MiniMax-M3.
- **Rounding steps:**
  1. The exponent is an fp32 quotient, `f32(2i) / f32(d)`.
  2. `base ** e` is one fp32 `pow`.
  3. The result is an fp32 reciprocal.
- **The helper** (`zig/src/core/rope_freq.zig`) does exactly that, with the `pow` computed in f64 and rounded once to fp32:

  ```zig
  const e: f32 = @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(dim));
  const p: f32 = @floatCast(std.math.pow(f64, theta, e));
  f.* = 1.0 / p;
  ```

  - `inverseFrequencies(theta, dim, out)` is the default rope; partial rotary means `dim` = the rotated width.
  - `proportional(theta, dim, rotated, out)` is Gemma 4's global rope: the first `rotated` pairs over `dim`, zeros after.
- **Why this form:**
  - Under torch's default (scalar) CPU path, transformers' words equal the helper's on **every one of the 18 configs** below.
  - torch's vectorized `pow` (AVX2 and AVX-512) is not always correctly rounded. Where it rounds 1 ulp low, the helper keeps the correctly rounded word. I checked the 5 such words with 60-digit decimal arithmetic.
  - Those cases depend on the tensor's length and the element's position. For example, `1e6^0.578125` is `0x4537eba3` in tensors shorter than 32 elements and `0x4537eba2` in a full 32-element AVX-512 chunk.
  - So "transformers' words" are not a pure function of the config. The helper's words are torch's wherever torch rounds correctly, which is all words on the scalar path.

| config (theta, d) | AVX-512 words ≠ helper | AVX2 words ≠ helper |
|---|---|---|
| 1e4, 256 | 1 (i 111) | 1 |
| 1e6, 128 | 1 (i 37) | 1 |
| 1e6, 512 | 3 (i 29, 148, 233) | 3 |
| 100, 36 / 1e4, 48 / 1e4, 56 | 0 | 1 each |
| all others | 0 | 0 |

- **Tests in `rope_freq.zig`:**
  - transformers' words as literals for four real configs, all the same on torch's default, AVX2 and AVX-512 paths:
    - theta 1e7 over 64 (Qwen3.5 / Flash Next / MOSS / Qwen3.8-27B);
    - 5e6 over 64 (Spark2.5 full layers);
    - 1e4 over 36 (Qwen-VL 72-wide tower heads);
    - Gemma 4's proportional global table (1e6, 64 of 256 over 512). This one uses the default-path words; the comment notes word 29's vectorized value.
  - A planted control: the fp32-`pow` form misses 15 of 32 words, by up to 5 ulp.

## Audit: every copy on main

Grep: `pow(` lines naming theta, base or rope, plus a wider search (literal thetas, `inv`/`freq` names, `exp`/`log` forms). "vs torch" is against transformers' words on torch's default path, which equal the helper's; "=" means word for word.

**Precision forms:**
- **A:** `1 / pow(f32, theta, f32(2i)/f32(d))`
- **B:** `f32(1 / pow(f64, theta, 2i/d))`
- **C:** `f32(pow(f64, theta, -2i/d))`
- **D:** the helper's
- **E:** D with the reciprocal in f64
- **F:** `1 / f32(pow(f64, theta, 2i/d))`
- **G:** `pow(f32, theta, -f32(i)/f32(d/2))`

| copy | family / backend | form | config(s) served | vs torch |
|---|---|---|---|---|
| `core/qwen_vl/grid.zig:47` | Qwen3.5-VL tower (shared) | A | 1e4, 36 | 8/18 differ, max 3 ulp |
| `vision/qwen_cuda.zig:438` | Qwen-VL tower, CUDA | A | 1e4, 36 (and 40 for 80-wide heads) | 8/18 max 3 (9/20 max 3) |
| `families/gemma4/vision.zig:313` | Gemma 4 tower, Metal | A | 100, 36 | 5/18 max 2 |
| `families/gemma4/cuda_tower.zig:151` | Gemma 4 tower, CUDA | A | 100, 36 | 5/18 max 2 |
| `families/gemma4/config.zig:136` | Gemma 4 text, Metal host table | B | sliding 1e4, 256; global 1e6, 512 (64 rotated) | sliding 43/128 max 1; global 21/64 max 1 |
| `families/gemma4/cuda_config.zig:136` | Gemma 4 text, CUDA twin (sliding used) | B | as config.zig | as config.zig |
| `families/gemma4/xpu_load.zig:126` | Gemma 4 text, XPU | B | as config.zig | as config.zig |
| `families/gemma4/cuda_forward.zig:41` | Gemma 4 global, CUDA | E | 1e6, 512 (64 rotated) | **=** (migrated) |
| `families/gemma4/cuda_dflash.zig:197` | Gemma 4 DFlash drafter, CUDA | B | 1e6, 128 (the fixture's theta) | 25/64 max 1 |
| `families/glm/vision.zig:347` | GLM-4V tower, Metal | F | 1e4, 32 | **=** (Metal-only, not migrated) |
| `families/glm/cuda_vision.zig:241` | GLM-4V tower, CUDA | A | 1e4, 32 | 4/16 max 2 |
| `families/glm_dsa/layer.zig:26` | GLM-DSA, Metal | B | 8e6, 64 (the config default; real theta comes from the checkpoint) | 9/32 max 1 |
| `families/dory/cuda_model.zig:225` | Dory, CUDA | A (theta cast to f32) | 1e4, 256; 1e7, 256 | 63/128 max 4; 72/128 max 5 |
| `families/spark25/config.zig:85` | Spark2.5 (shared Metal + CUDA) | D | 1e4, 256; 5e6, 64 (zeros past 32) | **=** (not migrated, see below) |
| `families/qwen3_5/cuda_train_load.zig:89` | Qwen3.5 learner, CUDA | G | 1e7, 64 | 15/32 max 5 |
| `families/flashnext/cuda_attn.zig:164` | Flash Next, CUDA | C | 1e7, 64 | 15/32 max 1 |
| `families/flashnext/cuda_prompt_step.zig:131` | Flash Next, CUDA | C | 1e7, 64 | 15/32 max 1 |
| `families/flashnext/cuda_batch_state.zig:414` | Flash Next, CUDA | C | 1e7, 64 | 15/32 max 1 |
| `xpu/load.zig:200`, `:262` | Qwen3.8-27B, XPU (two loaders) | C | 1e7, 64 | 15/32 max 1 |
| `xpu/drafter.zig:302` | Qwen3.8-27B drafter, XPU | B | 1e7, 128 | 18/64 max 1 |
| `families/qwen4/rotary_table.zig:8` | Qwen4 (one table, both backends) | C | 1e4, 64 (native-dense fixture); 1e4, 4 (tiny) | 10/32 max 1; tiny = |
| `families/moss_image/encoder.zig:33` | MOSS image text encoder | A (theta cast to f32) | 1e7, 64 | 15/32 max 5 |
| `families/step5/vision.zig:297` | Step5 tower | A | 1e4, 48 | 8/24 max 2 |
| `families/qwen_image/host.zig:35` | Qwen-Image (diffusers model) | B | 1e4, 16 and 1e4, 56 (axes 16/56/56) | 3/8 max 1; 17/28 max 5 |

**Not fp32 inverse-frequency words** (kept out of the table):
- `families/minimax_m3/vision.zig:261` and `cuda_vision.zig:219`: an f64 table (1e4 over 26) used for f64 angles. The two backends have the same source.
- `families/kolibri1/forward.zig:152`: f64 angles `p / theta^(2j/dh)` straight into cos/sin, no fp32 table.
- **Kernel-side** (no host table to audit):
  - Gemma 4's Metal global table, computed on the GPU by an MLX-`pow` kernel (`gemma4/kernels.zig:120`);
  - Qwen3.5/3.8's `q27_rope` (Metal) and `cuda_rope` (CUDA).
- **Test oracles and fixtures:**
  - `dory/ref.zig:217` (f64 reference);
  - `hip/kernel_tests/ops.zig:326` (an fp32 oracle);
  - `hip/kernel_tests/flashnext_attn.zig:68` (f64);
  - `qwen4/qsa_norm_reference.zig:39` (C form, fixture input);
  - `moss_image/host.zig:131` (a test's bands);
  - `core/qwen_vl/grid.zig:93` (a test).
- **Not RoPE frequencies** (sinusoidal timestep embeddings): `moss_image/host.zig:90`, `h3/modulation.zig:50`, `qwen_image/host.zig:92`.
- **YaRN:** `core/rope_yarn.zig:19` is the precedent; `flashnext/rotary_config.zig:19` is its byte copy, kept because its receipts hash it. Its exponent is an f64 quotient rounded to f32, not an f32 division. That gives the same result for the power-of-two `dim` 64 it uses.

**Backends of one family that disagree:**
- **GLM-4V tower:** Metal (F) equals torch, CUDA (A) differs on 4 of 16 words, by up to 2 ulp.
- **Gemma 4 global layers:** CUDA (`cuda_forward`, now the helper) equals torch, while XPU (`xpu_load`, B) differs on 21 of 64 words. Metal computes the table on the GPU (not auditable here); the `config.zig` host global table (B) differs too.
  - Sliding layers use B on all three backends (Metal host table, CUDA via `cuda_config`, XPU), so they agree with each other but differ from torch on 43 of 128 words.
- **Gemma 4 vision:** Metal and CUDA agree (both A, 5/18 off torch).
- **Qwen-VL tower:** grid and qwen_cuda agree (both A).
- **Qwen3.8-27B:** XPU's text table (C) and drafter table (B) each differ from torch. The CUDA and Metal paths compute in kernels, not audited.

## Migrated

- **`gemma4/cuda_forward.zig` global table** now calls `rope_freq.proportional(cfg.global_theta, cfg.global_dim, cfg.global_rotated, out)` (commit `3588ae156`).
  - **Evidence:** the new test `"the global table is core's proportional one, word for word the expression it replaced"` keeps the old expression. It checks all 256 words against the helper's table for Gemma 4's served config (theta 1e6, 512, 64 rotated; the same in the E4B, 12B, 31B and assistant fixtures).
  - **Planted check of that test:** replacing the old expression with form B makes it fail (`expected 1062845455, found 1062845456`).
  - **Gate:** `zig build test-gemma4-cuda-host` passes 54 of 54.

## Not migrated, and why

- **`spark25/config.zig:85`:** its words equal the helper's for both configs it serves, but the file is compiled in two module trees.
  - **The Metal tree:** `tensorfold.zig`, with no `core` import, via `spark25/metal_model.zig`.
  - **The CUDA tree:** `spark25.zig`'s module, which imports `core` by name but can't reach `../../core` by path.
  - **What it needs:** a build change, either `core` added to the Metal engine module or a CUDA-only twin. That's a maintainer call; I left it unchanged.
- **`glm/vision.zig:347`:** its words equal the helper's on its config, but it's a Metal file that builds only on macOS. I left it rather than make an unbuilt change. It would be the one-line switch to `inverseFrequencies(1e4, 32, &inv)` with an equality test, once `core` reaches it.
- **Every other copy in the table:** words differ from torch's, and so from the helper's, on at least one served config. Changing them changes numerics; that is a maintainer decision.
  - **Which side matches torch:** the helper/torch side in every case. The copies are off by 1–5 ulp: A and G (fp32 `pow`) worst, B, C and F by 1–5 ulp.

## Gates run

- `zig build test` (host): 6,044 of 6,077 tests pass, 30 skipped, **3 fail**, all in `test-cuda-host`:
  - `listener: accepted IPv4 and IPv6 TCP streams disable Nagle…`
  - two `drama_voice_tool` cleanup-permission tests

  The same 3 fail on the untouched base `ed2858298` (`test-cuda-host` there: 919 pass, 3 fail). They come from this sandbox (its network, and running as root), not from this branch.
- `zig build test-gemma4-cuda-host`: 54/54. `zig test zig/src/core/rope_freq.zig`: 5/5.
- `zig build test-spark25-host test-gemma4-config`: 108/109, 1 skipped.
- `zig fmt --check` on the changed files: clean. `git diff --check`: clean.
- `tools/zig/lean_check.py`: base 424, head 424, **0 new** (`tools/zig/lean_new.py`).

## What I could not run

- **Hugging Face:** the checkpoints' `config.json`s could not be downloaded, so the configs come from the repo's fixtures and pinned constants. For DFlash (1e6) and GLM-DSA (8e6), the theta is the fixture's or the default, not a downloaded config.
- **GPU paths:** transformers' words on GPU (CUDA `powf`) and the GPU-side tables (Gemma 4 Metal global, Qwen kernels).
- **Diffusers:** Qwen-Image's own rope init (diffusers) was not compared; its row compares against transformers' default form.
- **Metal and other GPU builds:** none; the one migrated file is covered by `test-gemma4-cuda-host` on the host.
