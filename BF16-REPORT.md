# BF16-CORE report: host fp32 → bf16 copies against `core/bf16_round.zig`

Branch `core-bf16-host`, from upstream main `9e8c368b0` (newer than `ed2858298`). Delete this file before any PR.

## Summary

- I found **157 host copies** of an fp32 → bf16 (or f64 → bf16) conversion in `zig/src`, `zig/tests` and the `zig/*.zig` roots. I ported each one verbatim into a scratch program and compared it with `core/bf16_round.zig` on **all 2^32 fp32 bit patterns** (ReleaseFast, 4 threads, 3 min 12 s for all of them).
  - The 60 inline `@truncate(@as(u32, @bitCast(x)) >> 16)` fills that make random test data are one spelling. I ported that spelling once and list the sites by file.
- **25 copies are identical to `bits()` / `round32()` on every input.** They are all spelled with an explicit NaN branch that keeps the payload and sets the quiet bit. One more copy, `qwen4/rounding_bound.zig`'s `bf64`, is an f64 → f64 copy of `core.bf16`. It is identical to it on all 2^32 fp32 values widened to f64 and on 2^26 random f64 words.
- **Migrated: 16 copies in 14 files, in 5 families** (core, Flash Next, GLM, DeepSeek, Qwen4).
- **Not migrated: 10 identical copies**, because of module reach: the file can't import `core`, or I can't build its tree here.
- **Everything else differs, and stays as it is:**
  - **112 copies differ on NaN only.** Most have no NaN branch. A NaN whose payload sits in the low 16 bits then rounds to ±Inf, and `0x7fff8000`–`0x7fffffff` / `0xffff8000`–`0xffffffff` round to −0 / +0.
    - 59 of them use a plain `+`. In Debug and ReleaseSafe, the 32,768 negative NaNs `0xffff8000`–`0xffffffff` overflow it and panic.
    - Other copies map every NaN to a fixed word: `0x7fc0`, `0xffff` (torch's CPU vector path) or `0x7fff` (PTX `cvt.rn`).
  - **19 copies differ on finite values too:**
    - Truncation helpers: 2,139,062,400 finite words differ.
    - `drama3/metal/quant_reference.zig`'s `roundBf16`: it overflows to max-finite instead of Inf, and gives NaN for ±Inf.
    - `tests/qwen27/embed.zig`'s `bf64`: it rounds wrong in the fp32 subnormal range.
- I found **no copy that rounds a finite value wrongly and also feeds weights, receipts or the wire.**
  - The finite differences are all in test oracles, test data, or truncation of values that are exact in bf16 by construction.
  - `drama3/metal/quant_reference.zig` is an FP64 oracle "never used for inference". Its overflow and ±Inf behaviour differ from core.
- **Part B:** 30 parse sites. Two families can disagree between backends on the same config.json: the Qwen3.5 target and its DFlash2 drafter (also FlashNext CUDA by design, and Nemotron-H on XPU). A core helper would fix the Qwen3.5 cases. Details at the end.

## How the audit ran

- I extracted the copies with a script:
  - every function of ≤ 14 lines whose body bit-casts a float and shifts or masks by 16;
  - every line with `0x7fff` / `0x7FFF` rounding;
  - every `@bitCast(...) >> 16` or `& 0xffff0000` outside such a function.
- I read through the hits and dropped look-alikes: hashes, sampling keys, `0x7fff_ffff` masks, and decoders (`<< 16`). I ported the 157 converters verbatim. Each port sits in its own namespace with the file's helpers it calls (`promote`, `fromBf16`, `bf`, `rig.f16Bits`).
  - One mechanical change, so that ReleaseFast stays defined: a plain `+` on the rounding line became `+%`.
  - Separately, I flag the copies where the plain `+` overflows: the 59 marked "plain `+`" below. A NaN check before the add means it can't overflow.
- **Reference:**
  - fp32 → u16 copies: `bits(x)`.
  - fp32 → fp32 round trips: `round32(x)`, compared by bits.
  - f64 copies: every fp32 value widened, compared with `round32` widened.
- For the 17 f64-input copies, I also ran 2^24 random finite f64 values with exponents −140 to 128. I compared them with `core.bf16(x)` (one rounding) and with `round32(@floatCast(x))` (via fp32).
  - 13 of them go through fp32 and equal `round32(@floatCast(x))` on every sample. They differ from one rounding in 118 of 2^24 samples, which is double rounding, as intended.
  - `qwen4` `bf64` equals one rounding on every sample.
  - `drama3`'s `roundBf16` differs from both (on 62,634 and 62,516 samples), all above fp32's range.
  - The two truncation copies (`sampling_identity_test.zig`, `qwen27/embed.zig`'s `bf64`) differ from both, as truncation does.
- **Classes:**
  - **identical:** no input differs;
  - **NaN only:** finite and ±Inf inputs all agree, and some NaN input differs (in payload, quiet bit or sign, or it turns into Inf or zero);
  - **finite:** some non-NaN input differs.
- **Examples:**
  - Each class lists the first two differing inputs and the last one, as `input → copy's output`.
  - For comparison, core gives `0x7fc0` for `0x7f800001` and `0xffff` for `0xffffffff`. For `0x00008001` core gives `0x0001`, and `0xff80` (−Inf) for `0xff7fffff`. The f64 outputs are shown as f64 bits.

## What was migrated

Each copy became an alias of the core function. The callers didn't change.

| family | commit | copies | import |
|---|---|---|---|
| core | `core: the kernel contract's toBf16 is bf16_round.bits, and a sweep keeps the replaced spelling` | `core/contract/contract.zig` `toBf16` | relative `../bf16_round.zig` (same module as core) |
| flashnext | `flashnext: the CUDA host glue's toBf16 and roundBf16 are core's bf16_round` | `toBf16` in `cuda_{mix,prompt_wt,attn,embed,qmm,ple,hc,gdn,moe}.zig`; `roundBf16` in `cuda_mix.zig` and `cuda_hc.zig` (→ `round32`) | `@import("core")` (module `cuda_launch.zig`; 5 of the files already imported it) |
| glm | `glm: the CUDA weights' and loader gate's toBf16 are core's bf16_round.bits` | `glm/cuda_weights.zig` `toBf16`, `tests/cuda/glm_loader_gate.zig` `bf16` | the `core` const both files already had |
| deepseek_v41 | `deepseek_v41: named's bf16Round is core's bf16_round.bits` | `deepseek_v41/named.zig` `bf16Round` (pub; also called from `forms.zig`) | `@import("core")` (every dsv41 module gets `core` from `dsv41_shared.zig`) |
| qwen4 | `qwen4: bf64 is core's bf16_round.bf16` | `qwen4/rounding_bound.zig` `bf64` (f64) | `@import("core").contract.bf16_round`: qwen4's `core` is the core root on CUDA but a facade on the host and Metal (`qwen4_core_host.zig`, `qwen4_core_metal.zig`), and all three export `contract` = `core/contract.zig`, which exports `bf16_round` |

**Evidence:**

- **The exhaustive run.** Each migrated copy is in class A below: 0 of 2^32 inputs differ. `qwen4` `bf64` is identical to `core.bf16` on all 2^32 fp32 values widened to f64, and on 2^26 random f64 bit patterns (`q4.zig`, 0 differ).
- **Tests kept in the tree.** The exhaustive run takes about 1.2 s per copy in ReleaseFast; under `zig build test`'s Debug it would take minutes. So the tree keeps a sampled test, with the old spelling verbatim:
  - **`core/bf16_round.zig`.** Test "the replaced copies equal bits() on every zero, subnormal, infinity, NaN and tie, and 1M random words" keeps the Flash Next / GLM / DeepSeek spelling and compares it with `bits()` on:
    - all 2^25 words with exponent 0 or 255 (every ±0, subnormal, ±Inf and NaN);
    - every tie `hi << 16 | 0x8000` with its neighbours `0x7fff` and `0x8001` (3 × 65,536 words);
    - 2^20 random words.

    It runs in 1.8 s in Debug, compile included. Core's own `contract.toBf16` was spelled exactly as `bits()` is, so it needs no copy.
  - **`qwen4/rounding_bound.zig`.** A test keeps the old `bf64` and compares it with core's, by f64 bits, on:
    - every fp32 tie and its neighbours, widened;
    - the fp32 edges (±0, the least subnormal, the largest subnormal, the least normal, the largest finite, ±Inf, a NaN);
    - 2^20 random f64 words.
- **Planted controls** for the core test (run on a scratch copy, not committed):
  - Dropping the NaN branch fails at the first NaN: expected `0x7fc0` (32704), found `0x7f80` (32640).
  - Rounding ties away from zero (`(u + 0x8000) >> 16`) fails at the first subnormal tie: expected 0, found 1.

## What was not migrated, and why

### Identical, but out of reach (module reach)

| copy | feeds | reach | the change it would take |
|---|---|---|---|
| `fabric/collective.zig:165` `toBf16` | wire: bf16 collective payloads | The fabric module (`zig/src/fabric/fabric.zig`) is created with no imports in `metal_targets.zig:500`, `flashnext_batch_host.zig:11`, `glm_expiry.zig:7` and `residency_host.zig:11`. A relative `../core/...` import leaves its root. | Give every fabric module a `core` import. In the Metal tree, `core` is the engine (`tensorfold.zig`), so that also needs `pub const bf16_round = @import("core/bf16_round.zig");` there. |
| `fabric/tp/reference.zig:13` `f32ToBf16` | host reference of the TP collectives | Same, plus the TP roots under `fabric/tp/` (`tp.zig`, `portable_test.zig`, `roce_multi_host.zig`). | As above, for the TP modules in `tp.zig`. |
| `living/affine.zig:10` `bf16of` | weights: affine scales and biases | The living module (`living/living.zig`) has no imports, and core imports living (`cuda.zig:25`), so a `core` import would be a cycle. | Pass `bf16_round` some other way, or move the helper into living. Not worth a cycle. |
| `core/affine4_host.zig:11` `bf16of` | weights: affine-4 scales and biases | It is reached by relative import from 28 roots, among them the Flash Next host roots (`zig/flashnext_*.zig`) and the Metal engine. Those roots also import the `contract` module, which owns `bf16_round.zig`. I tried the relative import and it broke 9 compiles: `file exists in modules 'contract' and 'root'` (and `'tensorfold'`). Reverted. | Those roots would need to reach core's files only through modules: `contract` or `core`. |
| `core/quant/convert.zig:34` `bf16` | weights: loader element conversion | Same roots, same error. Reverted. | Same. |
| `families/moss_image/host.zig:6` `bf16` | activations: host rotary and modulation tables | It is in the Metal engine (where `core` is `tensorfold.zig`, which doesn't export `bf16_round`) and in the CUDA module rooted at `moss_image/cuda.zig` (whose root directory doesn't contain `core/`). | `pub const bf16_round = @import("core/bf16_round.zig");` in `tensorfold.zig`, then `@import("core").bf16_round` works in both. |
| `families/glm/vision.zig:338` `bf16` | inputs: image pixels | Metal engine; on Linux, `residency_owner_tests.zig` and `glm_*_tests.zig` compile it but no step here runs its tests (I planted a failing test; it never ran). | A relative `../../core/bf16_round.zig`, as `gemma4/layer_input.zig` already does in the same tree. I didn't do it because I can't build the macOS tree here. |
| `core/qwen_vl/tower.zig:321` `bf16` | inputs: image pixels | Metal engine and `residency_owner_tests.zig`. Same as above. | Same, with `../bf16_round.zig`. |
| `families/qwen_image/host.zig:6` `bf16` | activations: host rotary, time embedding, modulation | Metal engine only. | Same. |
| `families/flashnext/vision_contract.zig:6` `toBf16` | vision contract oracle | Metal engine plus `flashnext_batch_tests.zig` / `flashnext_bound_family.zig`, which also import `contract` (the same two-modules error as `convert.zig`). No Linux step runs its tests. | Same as `affine4_host.zig`. |

### Differs on NaN only

There are 112 copies, in classes B1–B3 and N1–N5 below. None was migrated.

- The brief allows migrating a NaN-only copy that is a pure test oracle, but only one by one. I didn't take that option for any copy.
- **Most of these oracles are in GPU tests** (`tests/cuda`, `src/hip/kernel_tests`, the `*_check.zig` files). Those can't run here. Some of them compare against kernels that write a canonical NaN (`cvt.rn` gives `0x7fff`; the HIP oracles want `0x7fc0`), so core's payload-keeping NaN would be a different word.
- **The NaN-only copies that feed weights or other production values** must stay, per rule 3 of the brief:
  - weights: `qwen3_5/cuda_*weights.zig`, `gemma4/*_weights.zig`, `flashnext/pack.zig`, `xpu/load.zig`, `deepseek_v41/engram_rows.zig`;
  - inputs: the vision towers;
  - wire: `core/pair_rows_layout.zig`;
  - a model constant: `gemma4/{cuda_,}forward.zig`'s embed scale.
- **Changes worth considering, but not in this brief:**
  - The **59 copies with a plain `+` and no NaN branch** (marked below) panic in Debug and ReleaseSafe on the negative NaNs `0xffff8000`–`0xffffffff`. Production files among them:
    - `xpu/load.zig:182`;
    - `cuda/vit.zig:321`;
    - `gemma4/cuda_assistant_weights.zig:144`;
    - `gemma4/vision.zig:304`;
    - `step5/vision.zig:251`;
    - `deepseek_v41/draft/dspark.zig:80`;
    - `flashnext/ple_scale.zig:91`;
    - the two `embedScale` lines (whose input is a finite `sqrt`).

    For weights, a NaN in a checkpoint would panic in Debug.
  - **`nemotron/glue_math.zig`** maps every NaN to `0x7fff` on purpose (PTX `cvt.rn.bf16.f32`). **`deepseek_v41/engram_rows.zig`** maps NaN to `0xffff` on purpose (torch's CPU vector path). Both should stay as they are.

### Differs on finite values

There are 19 copies, in classes T1–T3, F1 and F2. None was migrated.

- **Truncation helpers (T1–T3)**, and the 60 inline truncation fills, make test data from values chosen exact in bf16 (`hip/kernel_tests/rig.zig`: "test inputs are chosen exact in bf16"), or from random floats where any bf16 will do. `core/quant/widen.zig`'s `bf16Of` truncates table values that are bf16 already.
- **`drama3/metal/quant_reference.zig:38` `roundBf16`** (f64; FP64 oracle for the quantized Metal gate):
  - The tie with 2^128 at the largest finite value goes to max-finite `0x7f7f`, not Inf: `0x7f7f8000` → `3.39e38`; core gives Inf. All of `0x7f7f8000`–`0x7f7fffff` and the negatives differ.
  - ±Inf in gives NaN (`0xff800000` → NaN), because `|Inf − Inf|` is NaN and both comparisons are false.
  - Above fp32's range, `@floatCast` gives Inf first.
  - These are 65,538 fp32 inputs: the overflow edge and ±Inf. All other finite inputs agree. It is a test oracle, so inference is not affected.
- **`tests/qwen27/embed.zig:21` `bf64`** (test oracle) rounds the f64 to 8 significant bits, then casts to fp32 and truncates. In the fp32 subnormal range, the rounding step is not bf16's subnormal step. 5,592,192 inputs differ, e.g. `0x00008001` → `0x0000`, where core gives `0x0001`.

## The audit table

"feeds": what the result goes to.

- **weights:** packed at load.
- **inputs:** pixels or tables uploaded per request.
- **activations:** host-computed values used in inference.
- **wire:** bytes another process or rank reads.
- **oracle:** a test or check reference.
- **test data:** inputs a test makes.

**Paths** are relative to `zig/`. `src/...` is engine code; `tests/...` and the `zig/*.zig` roots are tests and benches.

**Notes:**

- "plain `+`" means a rounding add without wrap and without a NaN branch before it. It panics on `0xffff8000`–`0xffffffff` in Debug and ReleaseSafe.
- "(truncation fills, 60 sites)" stands for 60 inline `@truncate(@as(u32, @bitCast(x)) >> 16)` (or `@intCast`) test-data lines:
  - `cli/cuda_kernel_checks.zig:14`, `core/{frags,simd_qmm,simd_qmm_bits,tile_qmm}.zig`;
  - `families/flashnext/{attn2,grammar,host_sample,pack_widen_test,rows_load,sample:329}.zig`;
  - `families/glm/cuda_qpack.zig`, `families/lfm2_moe/backend.zig` (2);
  - `families/nemotron/{kernels,pocket_fit,head_block,paged_test}.zig`, `families/qwen27/{dflash/execution_test,rope_test}.zig`;
  - `families/qwen3_5/{draw (4),paged_test}.zig`, `families/step5/backend.zig`, `families/deepseek_v41/engram_host.zig`, `families/minimax_m3/cuda_exl3.zig`, `families/glm/cuda_mxfp8.zig`;
  - `living/{pocket_fit,restart_fixture,pocket_quiet (2),pocket_quiet_legacy_test,host_test}.zig`, `native/qwen27_mask_fake.zig`, `core/quant/{mlx_checkpoint_test,mlx_geometry_test}.zig`;
  - and in `tests/`: `bf16_topk`, `core_lane_small`, `cuda/deepseek_v41/{scores_b_bench (2),stream_rb_bench,x3ld_epi_bench}`, `cuda/glm_mm_bench (2)`, `cuda/radio_tests (2)`, `cuda/shared_fork`, `cuda/window_bench`, `flashnext_run (2)`, `glm_sample_check`, `lfm2_bench (5)`, `qwen27/{dflash_conv (2),synthetic_graph,dflash_graph,attention_fused,attention_precision}`, `tree_round`, `xpu/nem_attn_dec`, `contract/metal_nemotron_pages`, `cuda/contract_qwen_pages (2)`, `kimi_k3/synth`.

  Most sites are in test blocks. The others make random inputs for checks: the runtime self-checks `core/frags.zig` `matches`, `core/simd_qmm.zig` `pairedExact`, `core/tile_qmm.zig` `twinExact`, and Flash Next `rows_load.zig` / `attn2.zig`, plus `cli/cuda_kernel_checks.zig` and `living/restart_fixture.zig`. For random inputs, any bf16 does. `minimax_m3/cuda_exl3.zig` and `glm/cuda_mxfp8.zig` refuse an inexact value first, so they truncate only exact values.
- "(bf16-valued masks …)" stands for the `& 0xFFFF_0000` lines in `deepseek_v41/sampling_check.zig:25` and `sampling_test.zig:164,249,382`. They make bf16-valued test logits.
- `tests/contract/nemotron_experts_fixture.zig`'s `bfSide` rounds up or down on purpose, for interval ends. It is not a nearest-even copy, so it isn't counted.

#### B1. Ties to even, no NaN branch (fp32 -> bf16 bits): 89 copies, differs on NaN only

- Finite or ±Inf inputs that differ: 0
- NaN inputs that differ: 12,582,846, of which 131,072 give a non-NaN (e.g. `0x7f800001` → `0x7f80`, `0x7f800002` → `0x7f80`, `0xffffffff` → `0x0`)

| copy | feeds | note |
|---|---|---|
| `gemma4_head_test.zig:12` | test data |  |
| `exl3_projection_bench.zig:127` | bench data |  |
| `src/xpu/fp64_gdn.zig:91` | FP64 reference check (oracle) | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `src/cuda/vit.zig:321` | inputs: pixels and rotary tables (lfm2, gemma4 CUDA towers) | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `src/core/pair_rows_layout.zig:95` | wire: seam frame elements (pair rows ABI) |  |
| `src/core/exl3_projection_gpu_test.zig:37` | test oracle |  |
| `src/core/affine_mm.zig:130` | test data (random x, scales, biases) | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `src/core/affine4.zig:20` | weights: affine q4 metadata |  |
| `src/families/qwen27/quant_rows_test.zig:13` | test data |  |
| `src/families/drama3/metal/quant_format.zig:70` | weights: packed expert metadata |  |
| `src/families/gemma4/cuda_assistant_weights.zig:144` | weights: assistant at load | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `src/families/gemma4/vision.zig:304` | inputs: image pixels | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `src/families/nemotron/train_check.zig:11` | GPU check oracle | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `src/families/deepseek_v41/draft/dspark.zig:80` | weights: twin drafter tables (`twin.zig`) | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `src/families/glm/eh_check.zig:15` | GPU check oracle | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `src/families/glm/attn_check.zig:13` | GPU check oracle | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `src/families/glm/launch_check.zig:11` | GPU check oracle | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `src/families/glm/cuda_kv8_ref.zig:7` | CPU format reference (oracle) | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `src/families/h3/modulation.zig:63` | activations: modulation tables |  |
| `src/families/lfm2_moe/dspark_test.zig:49` | test oracle | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `src/families/qwen3_5/cuda_draft_weights.zig:89` | weights: drafter at load |  |
| `src/families/qwen3_5/cuda_exl3_weights.zig:76` | weights: EXL3 small tensors at load |  |
| `src/families/qwen3_5/cuda_weights.zig:239` | weights at load |  |
| `src/families/flashnext/cuda_draft_affine.zig:4` | weights: proposal head |  |
| `src/families/flashnext/rotary_fixture.zig:6` | test fixture |  |
| `src/families/flashnext/ple_scale.zig:91` | weights: n-gram rows times weight_scale | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `src/families/flashnext/fsr_gpu_fixture.zig:143` | test fixture |  |
| `src/families/step5/vision.zig:251` | inputs: pixels and rotary tables | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/glm_rows_check.zig:19` | test oracle or data |  |
| `tests/moe_bench.zig:29` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/tile_qmm_check.zig:10` | test oracle or data |  |
| `tests/glm_rows_kda.zig:16` | test oracle or data |  |
| `tests/nemotron_sample_check.zig:64` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/glm_rows_classes.zig:40` | test oracle or data |  |
| `tests/affine_ref.zig:14` | test oracle or data |  |
| `tests/exlr8_ref.zig:14` | test oracle or data |  |
| `tests/qwen36_row_seam.zig:6` | test oracle or data |  |
| `tests/lanes_see.zig:10` | test oracle or data |  |
| `tests/sq_bench.zig:22` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/mm_bench.zig:25` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/core_lane_reg.zig:7` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/flashnext_widths_lib.zig:9` | test oracle or data |  |
| `tests/deltanet_marks.zig:51` | test oracle or data |  |
| `tests/row_widths_check.zig:20` | test oracle or data |  |
| `tests/row_projection_check.zig:9` | test oracle or data |  |
| `tests/core_lane_projection.zig:11` | test oracle or data |  |
| `tests/qwen27/embed.zig:15` | test oracle or data |  |
| `tests/qwen27/widths.zig:46` | test oracle or data |  |
| `tests/xpu/nem_fp64.zig:33` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/cuda/dense_rows_tests.zig:11` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/cuda/train_back_rig.zig:19` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/cuda/window_tests.zig:11` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/cuda/block_tests.zig:10` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/cuda/glm_mm_fixed_tests.zig:11` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/cuda/sample_tests.zig:19` | test oracle or data |  |
| `tests/cuda/qk_rows_tests.zig:12` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/cuda/lfm2_attention_tests.zig:36` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/cuda/norms_tests.zig:60` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/cuda/tree_tests.zig:10` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/cuda/tree_round_tests.zig:39` | test oracle or data |  |
| `tests/cuda/fn_pocket_tests.zig:65` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/cuda/moe_tests.zig:11` | test oracle or data |  |
| `tests/cuda/gdn_prep_tests.zig:11` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/cuda/flashnext_rig.zig:6` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/cuda/norm_rows_tests.zig:13` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/cuda/q27_back_rig.zig:21` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/cuda/libs_tests.zig:10` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/cuda/mxfp8_tests.zig:11` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/cuda/glm_synth.zig:171` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/cuda/exl3_tp2_tests.zig:333` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/cuda/quant_ref_tests.zig:73` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/cuda/contract_fn_pages.zig:181` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/cuda/modelopt_tests.zig:18` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/cuda/experts3_tests.zig:14` | test oracle or data |  |
| `tests/cuda/moe_glue_tests.zig:47` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/cuda/glm_index_check.zig:25` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/cuda/route_rows_tests.zig:12` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/cuda/exl3_candidate_tests.zig:435` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/cuda/deepseek_v41/mhcdec.zig:48` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/kimi_k3/cluster.zig:45` | test oracle or data |  |
| `src/hip/kernel_tests/nemotron/fused.zig:17` | test oracle (HIP) | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `src/families/flashnext/pack.zig:66` | weights: packs from the checkpoint |  |
| `src/xpu/load.zig:182` | weights: fp16 norms to bf16 at load (XPU) | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `src/families/flashnext/sample.zig:187` | test data (GPU sampling test logits) | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `src/families/flashnext/sample.zig:243` | test data (GPU sampling test logits) | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `src/hip/affine_gpu_test.zig:66` | test data |  |
| `src/hip/affine_gpu_test.zig:163` | test data |  |
| `tests/contract/rotary_sweep_norm.zig:59` | test oracle or data |  |
| `tests/flashnext_batch_sample.zig:39` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |

#### A. Identical on all 2^32 inputs: 25 copies, differs on none

- Finite or ±Inf inputs that differ: 0
- NaN inputs that differ: 0, of which 0 give a non-NaN

| copy | feeds | note |
|---|---|---|
| `src/fabric/collective.zig:165` | wire: bf16 collective payloads (fabric) |  |
| `src/fabric/tp/reference.zig:13` | host reference of the TP collectives (oracle; fabric) |  |
| `src/living/affine.zig:10` | weights: affine scales/biases Living Weights writes |  |
| `src/core/affine4_host.zig:11` | weights: affine-4 scales/biases packed on the host |  |
| `src/core/contract/contract.zig:39` | kernel-contract oracle (`c.toBf16` in ~20 contract tests) | **migrated** |
| `src/core/quant/convert.zig:34` | weights: loader element conversion (torch `.to(bfloat16)`) |  |
| `src/core/qwen_vl/tower.zig:321` | inputs: image pixels to the Metal tower |  |
| `src/families/moss_image/host.zig:6` | activations: host rotary and modulation tables |  |
| `src/families/deepseek_v41/named.zig:78` | weights: rank tensors repacked at load | **migrated** |
| `src/families/glm/vision.zig:338` | inputs: image pixels to the tower |  |
| `src/families/glm/cuda_weights.zig:66` | weights at load | **migrated** |
| `src/families/flashnext/cuda_mix.zig:24` | host check of the decode mixer (oracle) | **migrated** |
| `src/families/flashnext/vision_contract.zig:6` | vision contract oracle |  |
| `src/families/flashnext/cuda_prompt_wt.zig:343` | weights: prompt linears at load | **migrated** |
| `src/families/flashnext/cuda_attn.zig:25` | host check of layer-47 attention (oracle) | **migrated** |
| `src/families/flashnext/cuda_embed.zig:123` | weights: token table bytes for the GPU and the host dequant | **migrated** |
| `src/families/flashnext/cuda_qmm.zig:87` | host check of the lane matmul (oracle) | **migrated** |
| `src/families/flashnext/cuda_ple.zig:154` | weights: n-gram table rows | **migrated** |
| `src/families/flashnext/cuda_hc.zig:53` | host check of the hyper-connection glue (oracle) | **migrated** |
| `src/families/flashnext/cuda_gdn.zig:21` | host check of one-row GDN (oracle) | **migrated** |
| `src/families/flashnext/cuda_moe.zig:33` | host check of layer-0 experts (oracle) | **migrated** |
| `src/families/qwen_image/host.zig:6` | activations: host rotary, time embedding, modulation |  |
| `tests/cuda/glm_loader_gate.zig:431` | test data: synthetic checkpoint bytes | **migrated** |
| `src/families/flashnext/cuda_mix.zig:34` | host check of the decode mixer (oracle) | **migrated** |
| `src/families/flashnext/cuda_hc.zig:158` | host check of the hyper-connection glue (oracle) | **migrated** |

#### T1. Truncation (`bits >> 16`), as a function: 15 copies, differs on finite and NaN

- Finite or ±Inf inputs that differ: 2,139,062,400 (e.g. `0x00008001` → `0x0`, `0x00008002` → `0x0`, `0xff7fffff` → `0xff7f`)
- NaN inputs that differ: 8,388,606, of which 131,070 give a non-NaN (e.g. `0x7f800001` → `0x7f80`, `0x7f800002` → `0x7f80`, `0xffbfffff` → `0xffbf`)

| copy | feeds | note |
|---|---|---|
| `src/living/swiglu_test.zig:11` | test oracle or data |  |
| `src/hip/kernel_tests/rig.zig:131` | test oracle or data |  |
| `src/hip/kernel_tests/affine/data.zig:28` | test oracle or data |  |
| `src/core/quant/widen.zig:209` | test helper (truncation: inputs chosen exact in bf16) |  |
| `src/families/gemma4/cuda_backend.zig:561` | test helper (truncation: inputs chosen exact in bf16) |  |
| `src/families/nemotron/cuda_subset_batch_test.zig:15` | test oracle or data |  |
| `tests/glm_dsa_index_check.zig:13` | test oracle or data |  |
| `tests/nemotron_experts.zig:94` | test oracle or data |  |
| `tests/glm_dsa_bench.zig:101` | test oracle or data |  |
| `tests/nemotron_dense.zig:205` | test oracle or data |  |
| `tests/lfm2_fp64.zig:22` | test oracle or data |  |
| `tests/gemma4/no_tensor.zig:10` | test oracle or data |  |
| `tests/cuda/experts_rows_tests.zig:46` | test oracle or data |  |
| `tests/cuda/exl3_candidate_tests.zig:49` | test oracle or data |  |
| `(truncation fills, 60 sites)` | test data |  |

#### B2. Ties to even, no NaN branch, round trip (fp32 -> fp32): 6 copies, differs on NaN only

- Finite or ±Inf inputs that differ: 0
- NaN inputs that differ: 12,582,846, of which 131,072 give a non-NaN (e.g. `0x7f800001` → `0x7f800000`, `0x7f800002` → `0x7f800000`, `0xffffffff` → `0x0`)

| copy | feeds | note |
|---|---|---|
| `qwen35_draft_precision_host.zig:110` | test oracle | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/gemma4/glue.zig:87` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `src/hip/kernel_tests/flashnext.zig:14` | test oracle (HIP) | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `src/families/moss_image/encoder.zig:42` | activations: rotary cos/sin tables |  |
| `src/families/gemma4/cuda_forward.zig:30` | model constant: the embedding scale sqrt(hidden) | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `src/families/gemma4/forward.zig:193` | model constant: the embedding scale sqrt(hidden) | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |

#### B3. Ties to even, no NaN branch, f64 -> fp32 -> bf16 -> f64: 6 copies, differs on NaN only

- Finite or ±Inf inputs that differ: 0
- NaN inputs that differ: 8,454,016, of which 131,072 give a non-NaN (e.g. `0x7f800001` → `0x7ff0000000000000`, `0x7f800002` → `0x7ff0000000000000`, `0xffffffff` → `0x0`)

| copy | feeds | note |
|---|---|---|
| `tests/cuda/kolibri_ref.zig:18` | test oracle or data |  |
| `src/cli/cuda_precision.zig:18` | check-precision tool (oracle) | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/cuda/exl3_experts_f64_tests.zig:77` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `src/families/flashnext/vision_bounds.zig:20` | contract bounds (oracle) |  |
| `tests/cuda/flashnext_rig.zig:16` | test oracle or data | plain `+`: panics on 0xffff8000-0xffffffff in Debug/ReleaseSafe |
| `tests/flashnext_widths_lib.zig:112` | test oracle or data |  |

#### N1. NaN to the canonical 0x7fc0: 5 copies, differs on NaN only

- Finite or ±Inf inputs that differ: 0
- NaN inputs that differ: 16,646,143, of which 0 give a non-NaN (e.g. `0x7f810000` → `0x7fc0`, `0x7f810001` → `0x7fc0`, `0xffffffff` → `0x7fc0`)

| copy | feeds | note |
|---|---|---|
| `src/vision/qwen_cuda.zig:427` | inputs: image pixels (CUDA Qwen-VL tower) |  |
| `src/core/nvfp4_rows_test.zig:9` | test oracle |  |
| `tests/cuda/gdn_back_ref.zig:67` | test oracle or data |  |
| `src/hip/kernel_tests/ops.zig:16` | test oracle (HIP) |  |
| `src/hip/kernel_tests/tp.zig:15` | test oracle (HIP) |  |

#### N2. Exponent 0xff truncated (Inf and NaN keep their top half): 2 copies, differs on NaN only

- Finite or ±Inf inputs that differ: 0
- NaN inputs that differ: 8,388,606, of which 131,070 give a non-NaN (e.g. `0x7f800001` → `0x7f80`, `0x7f800002` → `0x7f80`, `0xffbfffff` → `0xffbf`)

| copy | feeds | note |
|---|---|---|
| `src/families/gemma4/dflash_weights.zig:143` | weights: drafter quantized at load |  |
| `src/families/flashnext/pack.zig:188` | weights: dequantized affine rows in packs |  |

#### N3. NaN to 0xffff: 2 copies, differs on NaN only

- Finite or ±Inf inputs that differ: 0
- NaN inputs that differ: 16,646,142, of which 0 give a non-NaN (e.g. `0x7f800001` → `0xffff`, `0x7f800002` → `0xffff`, `0xfffeffff` → `0xffff`)

| copy | feeds | note |
|---|---|---|
| `src/families/deepseek_v41/engram_shared_test.zig:17` | test oracle |  |
| `src/families/deepseek_v41/engram_rows.zig:48` | weights: Engram e4m3 lookup table |  |

#### T2. Truncation of an f64 through fp32 (mask 0xffff0000): 1 copy, differs on finite and NaN

- Finite or ±Inf inputs that differ: 2,139,062,400 (e.g. `0x00008001` → `0x0`, `0x00008002` → `0x0`, `0xff7fffff` → `0xc7efe00000000000`)
- NaN inputs that differ: 131,070, of which 131,070 give a non-NaN (e.g. `0x7f800001` → `0x7ff0000000000000`, `0x7f800002` → `0x7ff0000000000000`, `0xff80ffff` → `0xfff0000000000000`)

| copy | feeds | note |
|---|---|---|
| `src/core/lanes/sampling_identity_test.zig:22` | test data (truncation) |  |

#### N4. NaN to 0x7fff (cvt.rn.bf16.f32): 1 copy, differs on NaN only

- Finite or ±Inf inputs that differ: 0
- NaN inputs that differ: 16,646,142, of which 0 give a non-NaN (e.g. `0x7f800001` → `0x7fff`, `0x7f800002` → `0x7fff`, `0xffffffff` → `0x7fff`)

| copy | feeds | note |
|---|---|---|
| `src/families/nemotron/glue_math.zig:14` | host mirror of the glue kernels (cvt.rn.bf16.f32) |  |

#### F1. f64 rounded at bit 45, then fp32, then truncated: 1 copy, differs on finite and NaN

- Finite or ±Inf inputs that differ: 5,592,192 (e.g. `0x00008001` → `0x0`, `0x00008002` → `0x0`, `0x807fbfff` → `0x807f`)
- NaN inputs that differ: 8,388,480, of which 131,072 give a non-NaN (e.g. `0x7f808001` → `0x7fc1`, `0x7f808002` → `0x7fc1`, `0xffffffff` → `0x0`)

| copy | feeds | note |
|---|---|---|
| `tests/qwen27/embed.zig:21` | test oracle or data |  |

#### F2. Nearest of two neighbours in f64: 1 copy, differs on finite and NaN

- Finite or ±Inf inputs that differ: 65,538 (e.g. `0x7f7f8000` → `0x47efe00000000000`, `0x7f7f8001` → `0x47efe00000000000`, `0xff800000` → `0xfff8200000000000`)
- NaN inputs that differ: 16,777,214, of which 131,072 give a non-NaN (e.g. `0x7f800001` → `0x7ff8200000000000`, `0x7f800002` → `0x7ff8200000000000`, `0xffffffff` → `0x0`)

| copy | feeds | note |
|---|---|---|
| `src/families/drama3/metal/quant_reference.zig:38` | FP64 oracle for the quantized Metal gate |  |

#### A2. f64 one-rounding copy of core.bf16 (identical to it): 1 copy, differs on none

- Finite or ±Inf inputs that differ: 0
- NaN inputs that differ: 16,776,960, of which 0 give a non-NaN (e.g. `0x7f800001` → `0x7ff8000020000000`, `0x7f800002` → `0x7ff8000040000000`, `0xffffffff` → `0xffffffffe0000000`)

| copy | feeds | note |
|---|---|---|
| `src/families/qwen4/rounding_bound.zig:8` | contract bounds (f64 oracle) | **migrated** |

#### N5. NaN to the canonical 0x7fc0, f64 round trip: 1 copy, differs on NaN only

- Finite or ±Inf inputs that differ: 0
- NaN inputs that differ: 16,646,143, of which 0 give a non-NaN (e.g. `0x7f810000` → `0x7ff8000000000000`, `0x7f810001` → `0x7ff8000000000000`, `0xffffffff` → `0x7ff8000000000000`)

| copy | feeds | note |
|---|---|---|
| `tests/cuda/gdn_back_ref.zig:77` | test oracle or data |  |

#### T3. Truncation mask on fp32 values (0xffff0000): 1 copy, differs on finite and NaN

- Finite or ±Inf inputs that differ: 2,139,062,400 (e.g. `0x00008001` → `0x0`, `0x00008002` → `0x0`, `0xff7fffff` → `0xff7f0000`)
- NaN inputs that differ: 8,388,606, of which 131,070 give a non-NaN (e.g. `0x7f800001` → `0x7f800000`, `0x7f800002` → `0x7f800000`, `0xffbfffff` → `0xffbf0000`)

| copy | feeds | note |
|---|---|---|
| `(bf16-valued masks: deepseek_v41 sampling_check/sampling_test)` | test oracle or data |  |

## Gates

All gates ran on the branch head; the base runs used the untouched base `9e8c368b0`.

- **`zig fmt --check zig`:** clean.
- **`git diff --check 9e8c368b0..HEAD`:** clean.
- **Lean:** `python3 -I tools/zig/lean_check.py` gives 424 problems on the base and 424 on the head; `tools/zig/lean_new.py` reports **0 new**.
- **File length:** every touched file is under 600 lines. The largest is `glm/cuda_weights.zig`, at 586.
- **`zig build test`** plus the host steps of every touched family, in one run of 66 steps:
  - **DeepSeek:** `test-dsv41-{shared,cuda-host,device-host,state,forward,kv,draft,bounds-host}`, `test-draft-dsv41-host` and `compile-dsv41-{cuda,tool,server,runner,device-host,forward,kv,draft,sampler,shared}`.
  - **CUDA runner:** `compile-cuda-tests` (it compiles `tests/cuda/glm_loader_gate.zig`).
  - **Qwen4:** all 12 `test-qwen4-*` steps.
  - **Flash Next:** `test-flashnext-{batch,vision-contracts,rotary-bound,bound-probes,precise,proposal,pages-contract,proposal-pack,loader,contract,glue-contract,score}-host` and `test-rotary-sweep-host`.
  - **Gemma 4:** `test-gemma4-{load-parts,owners,contract,config,affine,prompt-head,cuda}-host`.
  - **GLM:** `test-glm-{window-contract,contract,scheduling}-host`, `test-glm-expiry` and `test-glm-gathers`.
  - **Living and residency:** `test-living{,-host,-shards-host}`, `test-merge` and `test-metal-residency{,-owners}-host`.
  - **Quant:** `test-quant-geometry-host`.
  - **Head result:** 664 of 671 steps succeed and 3 fail. The failing tests are below. Every one also fails on the base, run with the same 66 steps.

    | test | step | base |
    |---|---|---|
    | `listener`: accepted IPv4 and IPv6 TCP streams disable Nagle | test-cuda-host | fails |
    | `drama_voice_tool`: cleanup failure warns after publication… | test-cuda-host | fails |
    | `drama_voice_tool`: cleanup restores private directory permissions… | test-cuda-host | fails |
    | `families.glm.prefill_test`: chunk invariance gets the cap, the grid guard raises BadChunk | test-glm-scheduling-host | fails |
    | `residency_nemotron_tests`: native inventory includes nested prompt, MTP and learned buffers (crash) | test-metal-residency-owners-host | fails |
    | `residency_qwen27_tests`: native inventory and lazy owners… (crash) | test-metal-residency-owners-host | fails |
    | `residency_qwen27_tests`: reusable drafter frame buffers… | test-metal-residency-owners-host | fails |

  - **Base result:** the base run also failed 5 `recall_store_test` tests and 1 `late_system_test` test that pass on the head. Its last step (`test-dsv41-device-host`) also hit a full disk: `NoSpaceLeft`.
- **The two new tests run inside these steps:**
  - `src.core.bf16_round.test.the replaced copies equal bits()…` passes in the core test binary (109 passed, 1 skipped).
  - `src.families.qwen4.rounding_bound.test.the replaced bf64 equals core's…` passes in the Qwen4 family binary.

  I checked both by running the built binaries by hand.
- **Planted controls:** see "What was migrated".

**A failed first attempt**, kept here for the record: I first also migrated `core/affine4_host.zig` and `core/quant/convert.zig` with relative imports. The gate run then had 9 compile errors (`file exists in modules 'contract' and 'root'`, and `'tensorfold'`). I reverted both files before committing, and they are in the module-reach list.

## What I could not run

- **The macOS Metal build** (`metal_targets.zig` runs only when the target is macOS). So I did not touch the four Metal-only identical copies (glm/vision, qwen_vl/tower, qwen_image/host, flashnext/vision_contract).
  - None of my migrated files is in the Metal engine tree (`tensorfold.zig`). I checked with an import graph from every literal module root in the build files.
  - Qwen4 does reach the Metal build through `qwen4_core_metal.zig`. There it uses the same `core.contract` path it already used for `spec`.
- **GPU tests** (`tests/cuda` runner, HIP kernel tests, XPU), so I could not check how NaN behaves on any device. That is the reason I migrated no NaN-only oracle.
- **The scratch programs** (the verbatim ports, the harness, the f64 sampler, the qwen4 comparison) are not committed. They live outside the tree (`/home/user/scratch/bf/h`).
  - The exhaustive results per copy are in the tables above.
  - The rerun cost is about 3 min on 4 cores for all 157 copies, plus a 7-minute ReleaseFast compile.

## Part B (audit only, no code): `layer_types` and sliding-window parsing

**Scope:**
- I read every place in `zig/src` (outside `*_test.zig` and fixture literals) that reads a config's attention-layer pattern.
- I checked these points against the code myself:
  - `qwen3_5/cuda_geometry.zig:329`, `qwen3_5/config_checks.zig:14-26` with `config.zig:448-451`;
  - `qwen27/config.zig:319-328`;
  - `xpu/qwen27.zig:40,74-76`;
  - `gemma4/dflash_weights.zig:60-63`.
- **Keys read nowhere:**
  - `sliding_window_pattern`, `max_window_layers`, `attn_layer_period` and `attn_layer_offset`.
  - `use_sliding_window` is read only by `qwen27/dflash/config.zig:99`.
  - Qwen4 refuses any key not on its list (`qwen4/config.zig:93-104`).

Paths are relative to `zig/src/families/` unless they start with `xpu/`, `living/` or `cluster/`.

| file:line | family | backend | what it parses | key absent | unknown string |
|---|---|---|---|---|---|
| dory/config.zig:330-350 (window 312) | Dory | CUDA | `layer_types` == layers; sliding / full / mlp | error | reject |
| kolibri1/config.zig:173-188 (window 160) | Kolibri-1 | CUDA | `layer_types` == layers; full / sliding | error | reject |
| step5/config.zig:139,150 (`list()` 78-81) | Step5 | Metal | `layer_types` over layers + MTP; length only **≥ n** | error | reject |
| drama3/config.zig:97-106 | Drama3 | CUDA + Metal | `layer_types` == 40; full / linear | error | reject |
| lfm2/config.zig:66-69,100-106 | LFM2 | CUDA | `layer_types` (conv / full); the layer count **is** the array length | refusal (an empty array counts as absent) | refusal |
| lfm2_moe/config.zig:123-131 | LFM2-MoE | Metal | `layer_types` == pinned count, ≥ 1 attention | error | reject |
| lfm3/config.zig:144-151 (window pinned 256) | LFM3 | host admission | exact match to the fixed schedule | error | reject |
| gemma4/config.zig:248-252 | Gemma 4 | Metal | must equal `i % 6 == 5` → full | error | reject |
| gemma4/cuda_config.zig:248-252 | Gemma 4 | CUDA, XPU | textual copy of the Metal one | error | reject |
| gemma4/assistant_weights.zig:62-67 | Gemma 4 MTP | Metal | list == 4; window == 1024 | error | reject |
| gemma4/cuda_assistant_weights.zig:55-60 | Gemma 4 MTP | CUDA | same | error | reject |
| gemma4/dflash_weights.zig:60-63 | Gemma 4 DFlash | Metal | optional list == layers | **all sliding** | **becomes full** |
| gemma4/cuda_dflash.zig:64-67 | Gemma 4 DFlash | CUDA | same | all sliding | becomes full |
| spark25/config.zig:173-177 (window pinned 512) | Spark 2.5 | CUDA + Metal | exact `i % 4 == 3` | error | reject |
| flashnext/config.zig:270-275; MTP 170-177 | FlashNext | Metal, HIP | `layer_types` any order; full or sparse → sparse; the interval is not read | error | reject |
| flashnext/cuda_config.zig:155-169 | FlashNext | CUDA | interval **required** == 4; `layer_types` optional, but if given must equal `i % 4` exactly (`sparse_attention` refused); MTP only `["full_attention"]` | derived from the pattern (null counts as absent) | reject |
| qwen3_5/config.zig:112,116-121 | Qwen3.5-2B | Metal | interval == 4 and the exact pattern, length 24 | error | reject |
| qwen3_5/config_checks.zig:14-26 (via config.zig:379-392) | Qwen3.5 | HIP | `layer_types` any order; interval via `optionalInt`: absent → 4, **null / 0 / false → 0 → InvalidConfig even when `layer_types` is given** | absent **or null** → interval rule | reject |
| qwen3_5/cuda_geometry.zig:329-334 | Qwen3.5 | CUDA | `layer_types` any order == layers; interval never read | **error** | reject |
| qwen27/config.zig:319-328 | Qwen3.5 27B | Metal | `layer_types` any order; else the interval (absent → 4) | interval rule; **null `layer_types` → BadConfig** | reject |
| qwen27/dflash/config.zig:99-113 (window 2048 at 61) | Qwen3.5 DFlash2 | Metal | `use_sliding_window` must be true; every layer sliding | error | reject |
| qwen3_5/cuda_draft_config.zig:67,89 | Qwen3.5 DFlash2 | CUDA | only `sliding_window` in [16, 2048]; every layer gets a window | n/a | n/a |
| xpu/drafter.zig:17 | Qwen3.5 DFlash2 | XPU | nothing: `WIN = 2047` | n/a | n/a |
| xpu/qwen27.zig:74-76 | Qwen3.5 27B | XPU | nothing: `(i + 1) % 4 == 0` hard-coded (line 40 says "checked against config.json at load"; no code does) | n/a | n/a |
| qwen4/config.zig:201-213 | Qwen4 | Metal + CUDA | `layer_types` any order; if the interval is present, it **must agree** (the only cross-check) | error | reject |
| moss_image/encoder_config.zig:65-72 | MOSS-Image encoder | Metal | strict `i % 4 == 3` | error | reject |
| deepseek_v41/config.zig:149 | DeepSeek v4.1 | CUDA | `sliding_window` only; mode from `compress_ratios` | default 128, but the metadata check requires the key | n/a |
| minimax_m3/dspark.zig:73, dflash2.zig:73 | MiniMax-M3 drafters | Metal | `dspark` / `dflash2` `.sliding_window` > 0, required | error | n/a |
| minimax_m3/cuda_dspark.zig:42 | MiniMax-M3 DSpark | CUDA | `dspark/config.json` → `text_config.sliding_window` | **4096** | n/a |
| nemotron/config.zig:244-249,421-432 | Nemotron-H | CUDA + Metal | `hybrid_override_pattern` wins, else `layers_block_type` | error | reject |
| nemotron/xpu_config.zig:29-34 | Nemotron-H | XPU | `layers_block_type` only | parse error | **becomes attention** |

Related index-list parsers: `glm/config.zig:187-193` (0-based `full_attn_layers`), `kimi_k3/config.zig:110-118` (1-based; a missing key hits `.?`, a 0 entry overflows), `cluster/model.zig:121-128` (1-based `kda_layers`, bad entries skipped), `ling/config.zig:380` (`layer_group_size`), `living/families.zig:77` (only the array length).

**Can two backends of one family disagree on the same config.json?**

- **Gemma 4 main config:** no.
  - `config.zig` and `cuda_config.zig` differ only in line 1 and an import.
  - The assistant's `layer_types` blocks are identical too. Before that point, CUDA also admits `gemma4_assistant` (31B) where Metal refuses it.
- **Gemma 4 DFlash:** the pattern logic agrees, lenient rules included.
  - Metal's window is an `@intCast` from i64 (`dflash_weights.zig:49`), so a negative window panics in safe builds. CUDA returns BadDraftConfig.
- **Qwen3.5 target: yes.**

  | config.json | Metal qwen27 | HIP | CUDA | XPU |
  |---|---|---|---|---|
  | no `layer_types` | interval rule | interval rule | refused | `(i+1) % 4` |
  | `"layer_types": null` | BadConfig | as absent | refused | ignored |
  | `layer_types` given, interval 0 / null / false | accepted | InvalidConfig | accepted | ignored |
  | `layer_types` contradicts the interval | accepted | accepted | accepted | `(i+1) % 4` anyway |

- **Qwen3.5 DFlash2 drafter: yes.**
  - Metal requires `use_sliding_window: true`, every layer sliding, and a window of 2048.
  - CUDA reads none of the pattern keys and accepts any window in [16, 2048].
  - XPU uses 2047.

  A drafter with one full layer is refused on Metal and runs all-sliding on CUDA.
- **FlashNext: yes, but by design** (the CUDA kernels are fixed to the 48-layer `i % 4` layout). The cases:
  - absent `layer_types`: error on Metal and HIP, derived on CUDA;
  - `sparse_attention`: accepted on Metal and HIP, refused on CUDA;
  - absent interval: ignored on Metal and HIP, refused on CUDA;
  - any other valid pattern: runs on Metal and HIP, refused on CUDA;
  - MTP lists differ too.
- **Nemotron-H: yes.** XPU ignores `hybrid_override_pattern` and treats unknown blocks as attention, where the main parser refuses them.
- **MiniMax-M3 DSpark:** the backends read different files. A folder without the key runs at window 4096 on CUDA.
- **All other families** have one parser, or one shared by their backends.

**Recommendation.**

- **A core helper is worth it, mainly for Qwen3.5.**
  - **Shape:** `core/layer_pattern.zig`. Parse `layer_types` into kinds, with:
    - a per-family string → kind table (with aliases such as `full|sparse`, `conv`, `mlp`);
    - `== n` length;
    - an explicit fallback (`required`, `interval{key, default, null_is_absent}`, or `all(kind)`);
    - an explicit unknown-string rule;
    - optionally, qwen4's interval agreement check.
  - **What it would remove:** the real divergence in the Qwen3.5 target (null handling and falsy intervals across `qwen27`, `config_checks`, `cuda_geometry`) and in the DFlash2 drafter (CUDA checks nothing).
  - **A second function** would cover index lists, with a 0- or 1-based setting (GLM is 0-based; Kimi K3 and the cluster planner are 1-based).
- **Sites that could use it without any behaviour change:**
  - Dory (needs three kinds, not a bit set), Kolibri, Drama3, LFM2-MoE, `qwen3_5/cuda_geometry`, FlashNext `config.zig` (with aliases), Qwen4 (with the agreement check), both Gemma assistants.
  - The pinned-schedule validators would parse with it and then compare: Gemma `config`/`cuda_config`, Spark 2.5, `qwen3_5/config.zig`, the MOSS encoder, LFM3, `qwen27/dflash`.
- **Sites where it would change behaviour, unless an option keeps today's:**
  - `qwen27/config.zig` against `config_checks.zig`: pick one rule for null and falsy values.
  - `qwen3_5/cuda_draft_config.zig`: adopting Metal's rules refuses more drafters.
  - The XPU qwen27 and drafter would start reading config.json.
  - Step5: `≥` would become `==`.
  - LFM2: its count comes from the array.
  - Gemma DFlash's lenient rules, on both backends.
  - `nemotron/xpu_config.zig`.
- **Leave as is:** FlashNext CUDA's pinning, which is on purpose.
- **Small separate fixes:**
  - `kimi_k3/config.zig:117`'s unchecked `.?` and `- 1`;
  - `gemma4/dflash_weights.zig:49`'s `@intCast`;
  - the stale comment at `xpu/qwen27.zig:40`.
