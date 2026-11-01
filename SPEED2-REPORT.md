# SPEED2: faster candidates for LFM3-VL's vision attention on Iris Xe

Offline work: write, compile, check. No timing was done. Delete this file before any PR.

Branch `lfm3-vit-attn-candidates` from `b8e645f23` (`lfm3-vl-xpu`).
- `938e9680f` adds `vit_attn_q4` (A1).
- `bcf6629a2` adds `vit_attn_qr` (A2).
- The tests commit and this report come last.
- `vit_attn` and every other kernel are unchanged. The candidates are opt-in kernel names; nothing calls them.

**For the laptop:** build with `zig build xpu-tests -Dxpu -Dxpu-device=tgllp` and run `zig-out/bin/tf-xpu-lfm3_vit-test`. Measure **A2 `vit_attn_qr` first**. Both candidates take `vit_attn`'s arguments and grid (`(n + 31) / 32` × `hq` work-groups); only the local size changes, 128 instead of 64.

## Tools used

- **ocloc:** Ubuntu 24.04's `intel-ocloc` 23.43.27642.40 with `libigc1` 1.0.15468.25. GitHub (api and releases) returns 403 here, so the official compute-runtime release could not be fetched. The laptop's driver JIT-compiles the SPIR-V with **its own** IGC, so its ISA can differ from these numbers.
- **Options:** the build's own for this kernel: `-cl-std=CL3.0`, plus `-DTF_XPU_NO_DPAS` for tgllp (`zig/build/xpu.zig`). ISA and stats come from `IGC_ShaderDumpEnable=1` (`.asm`, `.zeinfo`).
- **CPU OpenCL check:** Intel's CPU OpenCL runtime (`intel-opencl-rt` 2026.2 and `intel-cmplr-lib-rt` from PyPI) with `pyopencl` 2026.1.4. It supports `cl_intel_subgroups` at sub-group size 16, so the real `.cl` file runs unchanged.
- **Zig 0.17.0** from the PyPI wheel. Helper scripts lived in my scratch folder and are not committed.

## 1. Paper bound

**Work** (A4 page, n = 8,580; 6 attention layers of 16 heads, head dim 64):
- **FLOPs:** 4·n²·64 per head = 18.8 GFLOP. Per layer that is 301.5 GFLOP; per page, **1.81 TFLOP**.
- **Time today:** at the measured 199 GFLOPS that is 9.1 s, which matches "~80 % of 11.2 s".
- **Floors:** at the fp32 FMA peak (1,681) **1.08 s**; at our GEMM's rate (763–841) **2.2–2.4 s**.
- **exp:** n² + n²/16 per head, about 7.5 G a page. That is small next to the FLOPs, even with IGC's multi-instruction `exp`.

**K/V bytes per layer** (reads from L3 or memory; unique K+V is 35 MB):

| form | K/V reads a layer | FLOP/B |
|---|---|---|
| `vit_attn` (each sub-group of 8 rows streams its head) | 1024·n² B = 75 GB | 4 |
| A1/A2 (4 rows a sub-group, no sharing) | 2048·n² B = 151 GB | 2 |
| shared by a 64-row work-group and the GQA pair | 64·n² B = 4.7 GB | 64 |

**Where `vit_attn` loses** (tgllp, ocloc 23.43, per sub-group per full 16-key tile):
- **Spills:**
  - Gen12LP has a fixed 128 GRFs of 32 bytes, and a SIMD16 float takes 2 GRFs. `o[8][4]`, `m[8]`, `l[8]` and `s[8]` alone are 112 GRFs.
  - ocloc reports **spill 896 B** (scratch 1,024 B a thread).
  - The P·V loop has **15 scratch spill/fill sends per key**, so about 282 scratch sends per tile (42 + 16 × 15).
- **SLM broadcast of q:** every 8-dim chunk of every row is a SIMD16 SLM read in which all 16 lanes load the same 32 bytes. That is 4 KB of SLM traffic per row per tile for 4,096 FLOP: **1 FLOP per SLM byte**.
  - If the SLM serves about 64 B/clk per subslice and does not merge same-address reads, the ceiling is about 64 × 6 × 1.1 GHz ≈ **420 GFLOPS**. This is an assumption: I can't check whether broadcasts are merged.
- **Variable-index shuffles:** in the P·V loop, `sub_group_shuffle(s[r], j)` with a variable `j` needs an indirect register read every time.
- **Issue:** about 3,190 dynamic instructions per tile for 8 rows, i.e. **398 a row against 128 `mad`s a row**. That is 32 % FMA density, an issue-bound ceiling of about 540 GFLOPS.
- **Occupancy:** 8 KB of SLM per 4-thread work-group gives about 32 of 112 threads a subslice, assuming 64 KB SLM a subslice on Xe-LP; please verify. Latency hiding is weak, and the spills add latency.
- **Hypothesis:** spills plus the SLM broadcast plus low occupancy explain 199 against the roughly 540 issue bound. Removing all three should get within reach of the issue bound.

## 2. Candidates

### A1 `vit_attn_q4`: class A (bit-equal)

- **Idea:** `vit_attn` with QR = 4 and 8 sub-groups, so the same 32 rows a work-group. The source is `vit_attn`'s, with only the row count changed.
- **ocloc (tgllp):**
  - SIMD16, 128 GRF, **spill 64 B** (scratch 1,024 B a thread, the same allocation as `vit_attn`), SLM 8 KB a work-group.
  - Key loop: 855 instructions, 64 SLM, 17 global and 1 scratch send.
  - P·V loop: **42 instructions, 16 `mad`, 0 scratch**.
  - About 1,485 dynamic instructions a tile for 4 rows, i.e. **371 a row**.
- **Expected: 1.2–1.8×.**
  - The spill sends are gone, and occupancy doubles to about 64 threads a subslice (8 threads per 8 KB).
  - SLM bytes per FLOP stay the same (1 B/FLOP), and instructions per row are about the same.
  - If the SLM broadcast is the limit, it stays near 1×.
- **Why measure it:** it's a diagnostic. A1 at about 1× means the SLM broadcast or issue is the limit, not the spills.
- **Risks:**
  - K/V reads per FLOP double (2 FLOP/B).
  - A 64-byte spill remains, outside the P·V loop.

### A2 `vit_attn_qr`: class A (bit-equal), recommended first

- **Idea:** A1's rows, plus three changes:
  - **Q in registers:** lane `l` holds dims 4l..4l+3 of each of its 4 rows (32 GRFs). For each 8-dim chunk, the 8 q values reach every lane through `sub_group_shuffle` with a constant index, which needs no SLM or memory traffic.
  - **32-bit offsets** from the kernel arguments. A stepped pointer made IGC lose the buffer and emit waterfall A64 gathers; argument plus offset keeps the stateful loads.
  - **Unrolled full-tile P·V:** a full tile runs `j` = 0..15 unrolled, so each `sub_group_shuffle(s[r], j)` is a fixed register region. The tail tile keeps the loop.
- **ocloc (tgllp):**
  - SIMD16, 128 GRF, **no spill, no scratch, no SLM**.
  - Key loop: 1,070 static instructions with **484 `mad`s**, 33 global sends, 0 SLM, 0 scratch. The 41-instruction tail loop does not run on full tiles.
  - About 1,029 dynamic instructions a tile for 4 rows, i.e. **257 a row**: 50 % FMA density, an issue bound of about 840 GFLOPS.
- **Expected: 2.1–3.0× (about 420–590 GFLOPS).**
  - All three losses above are gone, and occupancy is limited only by threads (112 a subslice, no SLM).
  - The realistic rate is 50–70 % of its issue bound.
  - On the A4 page, attention would go from 9.1 s to about 3.0–4.3 s, and the tower from 11.2 s to about 5–6.5 s.
- **Risks:**
  - K/V reads are 2 FLOP/B: about 250–300 GB/s from L3 at 500–600 GFLOPS. If Xe-LP's L3 can't serve that (I assume roughly 400+ GB/s), A2 becomes L3-bound. The fix would be the shared-tile idea (A3 below) once it compiles without a spill.
  - Code size grows with the unrolled tile (1,425 instructions in all).
  - The laptop's newer IGC may allocate registers differently; dump its ISA (below).

### A3 `vit_attn_kv`: written, compiled, **discarded** (spills)

- **Idea:** A2 plus 16-key K/V tiles staged in SLM and shared by a GQA pair's 2 heads × 4 row blocks (8 sub-groups):
  - K/V reads 8× fewer than A2 (16 FLOP/B);
  - 8 KB SLM per 8 threads;
  - K rows padded to 65 floats so each lane's read of its own key is conflict-free.
- **ocloc:** **spill 512–576 B** in every variant I tried: transposed K, padded rows, uniform staging indices, a fixed-length P·V loop. That leaves 4–8 scratch sends a tile, mostly predicate flags and staging addresses.
- **Status:** the rule says a spilling candidate is dropped unless fixed, so it is not in the branch.
- **Worth retrying:** only if A2 turns out L3-bound, and preferably first on the laptop's IGC.

### B (order change): evaluated on paper, not written

- **Idea:** 64-key softmax steps; each lane holds 4 keys, with one max/sum reduction and one `o` rescale per 64 keys.
- **What it changes:**
  - `m` is taken over 64 keys at a time;
  - `l` becomes a sum of 64 exps, reduced in a different order;
  - the `o` rescale factors `exp(m_old - m_new)` are applied 4× less often, so P·V partial sums are scaled at different points.
  - It would need the float64 gate (same or closer rel than `vit_attn`) in place of the byte gate.
- **Expected gain:** in A2's tile, the softmax part (reductions, two `exp`, the rescale) is about 269 of 1,066 instructions (about 25 %). Cutting it to a quarter saves about 19 %, so B ≤ 1.25× over A2.
- **Decision:** that is within 30 % of A2, so by your rule **A is preferred**, and I did not write B.

## 3. Exactness: why A1 and A2 give `vit_attn`'s bits (rnd = 0, and rnd = 1)

Each output row (row, head) is computed from its own state only (`m`, `l`, `o`, `s` of that row). The other rows of a sub-group never enter its arithmetic. So it is enough to show that each row sees the same operations on the same values in the same order.

1. **q values:**
   - `vit_attn` reads `q[(row·hq + head)·64 + d]` into SLM; A1 does the same.
   - A2 loads the same floats into registers and passes them through `sub_group_shuffle`, which copies bits.
   - Rows ≥ n: `vit_attn` uses zeros. A2 loads a clamped real row, then selects zeros. Those rows are never stored.
2. **Scores:**
   - Tiles `k0` = 0, 16, 32, … and lane ↔ key `k0 + lane` are unchanged.
   - Per row, `s[r] += dot(qv.lo, kv.lo) + dot(qv.hi, kv.hi)` runs for d = 0, 8, …, 56 in that order. The source text is the same, so the IR is the same: two `dot` calls, one add, one add into `s`, and no contraction across the calls.
   - `kv` holds the same K floats. A2 reads them at `k + (min(key, n−1)·hkv·64 + kvh·64) + d`. For key < n that is the same address as `vit_attn`'s. For key ≥ n, `s` is never read: `sc = -INF`, `pr = 0`, as in `vit_attn`.
3. **Softmax update,** per row per tile, statement for statement as in `vit_attn`:
   - `sc`
   - `mn = fmax(m, sub_group_reduce_max(sc))`
   - `pr`, `a`
   - `l = l·a + sub_group_reduce_add(pr)`
   - `m`, the four `o *= a`
   - `s = pr`

   The reductions see the same 16 values in the same lanes.
4. **P·V:** `j` = 0..kn−1 in increasing order; `vv` holds the same V floats (A2: a 32-bit offset to the same element); `pj = sub_group_shuffle(s[r], j)`; four `fma(pj, vv.c, o)` in the same order. Unrolling a full tile changes neither the operations nor their order.
5. **Output:** `il = 1 / l`, then `rd(o·il, rnd)` with the same `rd`. So rnd = 1 is bit-equal too.

**What the proof rests on:** IGC lowers identical source statements (`dot`, `exp`, `fma`, the sub-group reductions) identically in different kernels. That is expected (same builtins, no fast-math), but only the device can prove it. The test therefore checks bytes on the GPU.

**Evidence so far (CPU OpenCL, the real `.cl` file):**
- `vit_attn_q4` and `vit_attn_qr` give **byte-identical** output to `vit_attn` at n = 280, 1,156, 1,400 and 8,580, all 16 heads.
- Rel err to float64 is 7.2e-7 to 2.8e-6 (test tolerance 1e-5).
- The three planted controls are rel ≈ 1, so they fail as planted.
- Planted check of the check: in a copy of `vit_attn_qr`, `s += dot(lo) + dot(hi)` changed to `(s + dot(lo)) + dot(hi)` gives **DIFFER (1,090,709 of 1,183,744 words)**. A first attempt that only swapped the two operands of `+` stayed equal, as IEEE addition is commutative, so that one was no control.

## 4. Tests for the laptop (`zig/tests/xpu/lfm3_vit.zig`)

The attention block now runs `vit_attn`, `vit_attn_q4` and `vit_attn_qr` on the same inputs (n = 280 and 1,156; 16 query heads on 8 KV heads).
- **Each kernel:** rel err to float64 ≤ 1e-5, with three planted controls that must fail:
  - a causal mask;
  - KV head h % 8 (the swapped GQA mapping);
  - q's rows read transposed (`[head][row]`). This control is new, and `vit_attn` gets it too.
- **Each candidate:** its output bytes must equal `vit_attn`'s, printed as `... bytes against vit_attn's, n: same`, or `DIFFER (FAIL)`.

What I ran:
- `zig build xpu-tests -Dxpu -Dxpu-device=tgllp`: **succeeds**, with ocloc producing every SPIR-V image. It builds all 27 `tf-xpu-*-test` programs, including `tf-xpu-lfm3_vit-test`.
- Running `tf-xpu-lfm3_vit-test` here stops at `error: LoaderUnavailable` (no Level Zero, no GPU), as expected.
- Also: `zig fmt --check`, `zig ast-check`, and `git diff --check` all clean. `tools/zig/lean_new.py`: base 424, head 424, **0 new**.
- Every existing kernel's ISA from ocloc is **identical** to the base: `vit_attn`, `vit_gemm`, `vit_rms`, `vit_add_rms`, `vit_layernorm`, `vit_gelu`, `vit_dwconv`, `vit_qk_norm_rope`, `vit_swiglu`, compared by the SHA-256 of their instruction text.

## 5. Recommendation

1. **`vit_attn_qr` (A2) first.** It removes the spill, the SLM broadcast and the indirect shuffles at once, and has the best issue bound (about 840 GFLOPS). Expected 2.1–3.0×.
2. **`vit_attn_q4` (A1) second,** as the diagnostic. If it is about 1×, the SLM broadcast (or issue) was the limit; if it is about 1.5×, the spills were. That tells whether to go on to K/V sharing (A3) or to a B order change.

To time them: same arguments as `vit_attn`, grid `(n + 31) / 32` × `hq`, local size 128 instead of 64. Before timing, please also dump the ISA on the laptop's driver (`IGC_ShaderDumpEnable=1`, look for `spill size` in the `vit_attn_qr` `.asm`). Its IGC is newer than ocloc 23.43's, and a spill there would change the ranking.

## What I could not check

- **On the GPU:** no timing, no run, and no byte equality on the device. Equality was shown only on Intel's CPU OpenCL.
- **The laptop's compiler:** its IGC version and ISA (spills, GRF use) may differ from ocloc 23.43's.
- **Newest ocloc:** the official compute-runtime release could not be fetched (GitHub 403).
- **Hardware constants assumed:** SLM size (64 KB a subslice), SLM and L3 bandwidth (about 64 B/clk a subslice), and whether SLM merges same-address reads. These change the expected ranges, not the bit-equality.
- **A3 on a newer IGC,** and B: B was evaluated on paper only.
