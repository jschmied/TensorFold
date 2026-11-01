# Cloud report: XPU host table fix, LFM3-VL vision spec

CPU-only cloud session. Delete this file before any PR.

## Tools

- **Zig 0.17.0:** ziglang.org is blocked by this environment's network policy (403 on CONNECT). I used the PyPI wheel `ziglang==0.17.0` (manylinux x86_64, sha256 `97079827064a7492cce5541ceb438ae8003ce1e54e399676da2d62fb7a5dc08b`), which ships the official Linux binary. `zig version` prints `0.17.0`.
- **Python:** 3.13.16, every check run with `python3 -I`.
- **Hugging Face:** `huggingface.co` stayed blocked (403) for the whole session, also after the allow-list change. That change probably only reaches new sessions. So the real index was **not** downloaded; see "What I could not check".

## Task A: `test-slide-xpu-host` compiles again

- **Branch:** `fix-slide-xpu-host-table` from `ashhart/TensorFold` main at `7651ce64f`, one commit `5d26ca070`, "build: the XPU host tests' empty kernel table names every image the engines read".
- **Cause:** the empty `draft_xpu_images.zig` table in `zig/build/xpu.zig` (`hostTests`) is imported as both `xpu_qwen27_kernels` and `xpu_gemma4_kernels`. It had every Qwen 27B member but not three of the Gemma 4 table's (`xpu.zig` lines 84–93): `rows`, `rows_g`, `ngen_geglu`.
- **Fix:** add those three, so the empty table holds the union of both real tables (Qwen: glue, decode, gemm, gemm128, gdn, drafter, ngen_wide/narrow(128), ngen_swiglu(128), ngen_a8, ngen_a8_swiglu, ngen_a4; Gemma 4: glue, glue_g, rows, rows_g, gemm, ngen_wide, ngen_narrow, ngen_geglu). Checked against every `kb.*` use under `zig/src/xpu` and `zig/src/families/gemma4`.
- **Before:** `zig build test-slide-xpu-host test-draft-xpu-host` failed with `zig/src/families/gemma4/xpu_forward.zig:147:39: error: root source file struct 'draft_xpu_images' has no member named 'rows'`.
- **After:** `Build Summary: 8/8 steps succeeded; 4/4 tests passed` (test-slide-xpu-host 2 pass, test-draft-xpu-host 2 pass). No GPU, no ocloc. `zig fmt --check zig/build/xpu.zig` and `git diff --check` clean.

## Task B: one vision table and one image-row layout

- **Branch:** `lfm3-vision-spec` on `pull/1502/head` (`f066644fa`).
  - `facc1f878` "lfm3: vision_spec, …": `vision_spec.zig`, `vision_spec_test.zig`, and one line in `zig/build/lfm3_host.zig` (the test joins `test-lfm3-host`).
  - `365a5ab80` "lfm3 metal: …": the Metal loader and image rows read the spec.
  - This report is the last commit.

### `zig/src/families/lfm3/vision_spec.zig` (imports only `std` and `config.zig`)

- **Geometry:** `Vision` holds what the backends read from `vision_config`: hidden, intermediate, layers, heads, kv_heads, channels, kernel (`conv_kernel_size`), patch, temporal, merge (`spatial_merge_size`), plus `projector_hidden_size` and the text `hidden_size`, and each layer's mixer from `layer_types` (`conv` / `full_attention`).
  - Derived: `headDim()` 64, `kvWidth()` 512, `patchWidth()` 1536, `merged()` 4096, `tokens(gh, gw)` (one row per merge block).
  - `read(gpa, bytes)` parses config.json. It refuses a missing field, an unknown mixer, `downsample_factor != spatial_merge_size`, or a `head_dim` that disagrees with hidden / heads. Admission (which geometry the engines run) stays in `config.zig`.
  - `lfm3_vl_2b` is the one geometry `config.zig` admits, with `text_hidden = config.hidden`.
- **Tensor table, in `names.zig`'s style:**
  - a `Module` enum with `prefix()`, `layered()` and `path()`; `name(buf, layer, m)` builds the full name.
  - `shape(v, m)` gives each module's stored shape; `tensor(buf, v, layer, m)` returns name, shape and dtype (bf16), and refuses a module the layer does not hold.
  - `table(a, v)` lists all tensors in load order; `count(v)` is 138 for the 2B.
  - `check(table, keys)` passes when the vision keys are exactly the table (decoder keys are ignored). It fails with `MissingLfm3VisionTensor` or `UnexpectedLfm3VisionTensor`.
  - The spatial conv is `[1024, 3, 3, 1]`, MLX's layout as stored. The CUDA loader also accepts torch's `[1024, 1, 3, 3]`; the spec names the MLX one.
- **Image rows, no Metal type:**
  - `Span{at, rows, source}` (moved from `image_rows.zig`).
  - `overlap(span, at, rows) ?Copy{source, row, rows}`: the arithmetic `image_rows.zig` used.
  - `place(out, images, from, to, capacity)`: picks the pictures a prompt chunk reads and packs their rows from image row 0, the loop in `native/lfm3_images.zig` and XPU's `Engine.pictures`. Errors are `ImageTooLarge` and `TooManyImages`.

### Tests: `vision_spec_test.zig`, part of `test-lfm3-host`

- `config_test.vl` reads to exactly `lfm3_vl_2b`, plus the derived widths.
- Three planted config breaks are refused: a missing `num_key_value_heads`, a `sliding_attention` mixer, `downsample_factor` 4.
- **Table:**
  - 138 entries with unique names.
  - Ten names and shapes match what the Metal, XPU and CUDA loaders expect (patchifier, conv in_proj, spatial conv, k_proj 512, q_layernorm 64, w2, linear_1 4096×4096, linear_2 2048×4096, …).
  - `tensor` refuses a wrong module or layer.
- **Planted controls:** a renamed vision tensor fails, a missing one fails, an extra `multi_modal_projector.linear_3.weight` fails, and a decoder key passes.
- **Real index (`TF_LFM3_VL_DIR`):** reads that directory's `config.json` and `model.safetensors.index.json`. The index's vision keys must equal the table, and then one planted rename must fail. It skips when the variable is unset.
- **Image rows:** overlap at chunk edges; rows summed over all chunks equal the span; `place` packs in order, skips pictures before `from`, and refuses an over-budget request or a full `out`.

### Metal change (`365a5ab80`, mechanical, **unbuilt on macOS**)

- **`vision_weights.zig`:**
  - `width`/`intermediate`/`depth` come from `spec.lfm3_vl_2b`.
  - `Loader.get(layer, module)` takes the name and shape from `spec.tensor`; the dtype, byte and offset checks are unchanged.
  - Calls are the same and in the same order. The mixer test is `vision.mixers[i] == .attention` (the same layers as `i >= 2 and i % 2 == 0`), the prefix test is `spec.isVision`, and 138 is `spec.count(vision)`.
- **`image_rows.zig`:** `Span = spec.Span`, and `Images.encode` takes its copy range from `spec.overlap`. The bounds checks and the dispatch are unchanged.
- **Not changed:** `vision.zig`'s hard-coded widths (1536, 1024, 512, 2560, 4096, 2048) and `image.zig`'s constants. These are kernels' launch shapes; a later step can alias them to the spec.
- **Stub compile, Linux only:** I compiled the two changed files against stub `metal`, `sys`, `lanes` and `kernel_sources` modules, as an object whose root references `vision_weights.load` and `image_rows.Images.encode`. It is clean. As a control, planted typos `.w9` and `part.rowz` gave the two expected errors. The harness was not committed. A real macOS build and a VL run are still needed.

### Gates run on `lfm3-vision-spec`

- `zig build test-lfm3-host test-lfm3-image-host test-lfm3-prefill-host`: `15/15 steps succeeded; 14/15 tests passed (1 skipped)` on the final tree. The skip is the real-index test. The first, uncached run of the same steps showed `74/75 tests passed (1 skipped)`: image-host 34 pass, the config_test+vision_spec_test module 14 pass and 1 skip, the others 8, 6, 1 and 11 pass.
- `zig test -lc zig/src/families/lfm3/vision_spec_test.zig`:
  - without the variable: `14 passed; 1 skipped`
  - with `TF_LFM3_VL_DIR` pointing at an index **I wrote myself** from the table plus two decoder keys (a smoke run of the code path, not the real checkpoint): `All 15 tests passed`
  - with one name in that file renamed: `1 failed` (`UnexpectedLfm3VisionTensor`)
- `zig fmt --check` on the five changed files: clean. Zig 0.17's formatter rewrote `@intFromEnum` to `@backingInt` in `vision_spec.zig`.
- `python3 -I tools/zig/lean_check.py`: base `f066644fa` 424 problems, head 424; `tools/zig/lean_new.py base.txt head.txt`: `0 new`.
- `git diff --check`: clean.
- **Cross-check on the other branches (nothing pushed):** `facc1f878` cherry-picks cleanly onto `lfm3-vl-xpu` (`be4b868ef`) and `lfm3-vl-cuda-tower` (`b8e8d825d`). There `zig build test-lfm3-host` passes: XPU `28/29 (1 skipped)`, CUDA `55/56 (1 skipped)`.

### What the XPU and CUDA loaders should change (not done; those branches are untouched)

Step 1 for both: cherry-pick `facc1f878` (it applies cleanly, see above), then `const spec = @import("vision_spec.zig"); const vs = spec.lfm3_vl_2b;`.

**XPU (`lfm3-vl-xpu`, `be4b868ef`):**
- **`zig/src/families/lfm3/xpu_vision.zig`:**
  - Lines 9–17 (width, intermediate, depth, patch_width, out_width, heads, kv_heads, head_dim, proj_hidden) become aliases of `vs` (`vs.hidden`, `vs.intermediate`, `vs.layers`, `vs.patchWidth()`, `vs.text_hidden`, `vs.heads`, `vs.kv_heads`, `vs.headDim()`, `vs.projector_hidden`), cast to `u32` where the file uses `u32`. Keep `eps`/`ln_eps` (lines 18–19); the spec has no epsilons.
  - `fn get` (lines 67–76) becomes `get(l, seen, layer: ?usize, m: spec.Module)`: `const t = try spec.tensor(&b, vs, layer, m);`, then `t.name`, with `t.bytes()` as the length check. Keep the `"BF16"` dtype check.
  - In `Tower.load`, lines 113–121 and 142–157 become `get(l, &seen, null, .patchifier)` … `get(l, &seen, i, .k_norm)`. Delete lines 139–140 (the `vision_tower.layers.{d}.` prefix buffer). Line 147's `i >= 2 and i % 2 == 0` becomes `vs.mixers[i] == .attention`; line 161's `138` becomes `spec.count(vs)`.
- **`zig/src/families/lfm3/xpu_engine.zig` `pictures` (lines 95–110):** the skip, budget and count checks can come from `spec.place(&spans, imgs, from, std.math.maxInt(usize), e.image_tokens)`. The tower then runs once per placed span, at `span.source * out_width * 2`. `place` needs a picture's `.at` and `.tokens`, which the stream's images have.
- **`zig/src/families/lfm3/xpu_model.zig`:** line 78's `Image{rows, at, n}` can keep its device buffer. The splice at lines 487–492 (`lo`/`hi`) becomes `if (spec.overlap(.{ .at = im.at, .rows = im.n, .source = 0 }, p0, n)) |c| copy(from(s.x, c.row * H * 2), from(im.rows, c.source * H * 2), c.rows * H * 2)`.

**CUDA (`lfm3-vl-cuda-tower`, `b8e8d825d`):**
- **`zig/src/families/lfm3/vision_image.zig`:** lines 7–14 (hidden, heads, kv_heads, layers, intermediate, patch, temporal, merge) become aliases of `vs`. Its `marks`/`processor` stay.
- **`zig/src/families/lfm3/cuda_vision.zig`:**
  - Line 14 `patch_width` becomes `vs.patchWidth()`, line 16 `out_width` becomes `vs.text_hidden`, line 17 `merged` becomes `vs.merged()`, line 18 `projector_hidden` becomes `vs.projector_hidden`.
  - Delete line 23 (`attends`); use `vs.mixers[i] == .attention`.
  - In `load` (lines 127–170), every `name(...)`/`pv ++`/`pp ++` string and shape literal comes from `spec.tensor(&b[k], vs, i, .q_proj)` and its `.name`/`.shape.slice()`. `stacked` keeps its signature: pass the names and `shape.slice()`s of the stacked members (`.q_proj, .k_proj, .v_proj` and `.w1, .w3`). Delete the local `name`/`one` helper structs (lines 135–144) and `attends` in line 134's loop header once all call sites use the spec.
  - Keep the spatial conv's two-layout check (lines 156–157) or add torch's `[H, 1, 3, 3]` to the spec as an accepted alternative. That is a question for whoever owns the CUDA tower.
  - The CUDA branch has no image splice yet; when it gets one, use `spec.place` and `spec.overlap`.

## Task C: not done

`tools/lfm3/ref64.py` needs the LFM3-1.7B-MLX-4bit weights from Hugging Face, which this session cannot reach. No `lfm3-ref64-bulk` branch was made.

## What I could not check

- **The real `model.safetensors.index.json` and `config.json` of `LiquidAI/LFM3-VL-2B-MLX-4bit`:** not downloaded. `huggingface.co` returned 403 all session. The revision to pin is `c823fade98e35a3d6942ad4d38b1ea15f675cd12`: #1502's receipts (e.g. `docs/receipts/lfm3/local-standard-phase1.json`) and the fork's `docs/recipes/lfm3-native.md` ("VL 4-bit") name it.
  - To run it: download the two files into a directory, then `TF_LFM3_VL_DIR=<dir> zig test -lc zig/src/families/lfm3/vision_spec_test.zig` (the build step caches its run and does not see the variable).
  - The 138 names in the table equal what #1502's Metal loader already requires, all present and nothing extra. That loader ran on the real checkpoint in #1502's receipts, so I expect a pass, but it is unverified here.
- **Shapes against real headers:** the index holds names only. Shapes are checked by each loader at load time, as before.
- **Metal:** no macOS build and no LFM3-VL run of the changed loader or image rows. Only the Linux stub compile above.
- **Task C:** see above.
