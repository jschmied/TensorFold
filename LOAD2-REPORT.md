# LOAD2 report: Flash-Next CUDA load, the two levers left after #1873

Follow-up to ashhart/TensorFold#1901 ("Left out" in its description). Host only: no GPU, no nvcc here.

## Summary

- **Base:** fork branch `core-direct-gather` at `6eff834ab` (#1901's head, on main `58375fe00`). I did not push to it.
- **Pushed:** two branches, each one commit, authored and committed by Jürgen Schmied, with no trailer. No PRs.

| branch | SHA | files |
|---|---|---|
| `flashnext-store-use-keeps-open` | `b0ac4ee80` | `flashnext/cuda_prompt_wt.zig`, `flashnext/cuda_prompt_wt_test.zig` |
| `flashnext-gather-overlaps-pack` (on branch 1) | `7e3b1b726` | `flashnext/cuda_modelopt_threads.zig`, `flashnext/cuda_modelopt_threads_test.zig` |

Paths are under `zig/src/families/`.

- **Expected gain:** about 5.5–9.5 s off the 48 s cold load (to about 39–42 s). Branch 1 gives 2.5–4 s and branch 2 gives 3–5.5 s; the reasoning is at the end.

## Branch 1: `flashnext-store-use-keeps-open` (`b0ac4ee80`)

### Design

1. `use()` no longer switches shards with `closeFile()` + `Shard.load`.
2. It takes the next shard from `released` (the set `hold` / `release` keep) with `swapRemove`, or opens it once (`openShard`, now shared with `hold`).
3. The current shard is parked in `released` instead of closed. Its pages are dropped by the rule `Shard.close` uses: `File.release()` unless the shard holds the n-gram table (`runtimePages`). So the `use()` path drops exactly the pages it dropped before, and #1884's exemption is kept on this path (see the review note below).
4. `loaded()` and `deinit` close everything in `released` and the current shard, as before. After `loaded()` nothing is held, released or current.
5. Lifetimes: borrowed `tensor()` bytes are still documented as valid only until the next `use()` of another shard. The mapping now stays until `loaded()`, but no caller is told it may rely on that. The unused `path` field is gone, and the file is 588 lines.

**Edge case, unchanged in kind:** a shard can be open twice if `hold()` borrows from the shard that `tensor()` is currently on (`hold` does not look at the current shard, as before). Both copies close at `loaded()`.

**A review note on #1901:**
- The maintainer's point 1: `release()` drops pages for every borrower, bypassing #1884's n-gram exemption. That is #1901's code, which this task says not to touch, so I left `release()` as it is.
- Branch 1 does not widen it: a `use()` switch keeps the exemption. But if branch 1 is merged before point 1 is fixed, a shard parked by `use()` and later taken by `hold()` gets its pages dropped by `release()`.
- Fixing `release()` the same way (`if (!runtimePages(...))`) would cover both.

### Test and planted control

- **New test** "tensor() switching across three shards opens and parses each once a load, sharing the store's released shards" (`cuda_prompt_wt_test.zig`):
  - three shards, one tensor each;
  - after each shard's first `tensor()`, its file is deleted, so any second open or header parse fails with `FileNotFound`;
  - then four rounds of alternating `tensor()` across the three, with a `copyOf` in one round;
  - `hold()` takes a parked shard, `release()` gives it back, and `tensor()` takes it again;
  - after `loaded()`: no current shard, nothing held, nothing released.
- **Now:** OK.
- **Planted control:** the old `use()` with the new test fails at the first switch back:
  `error.FileNotFound` from `core.safetensors.File.open`, through `Store.use` → `Shard.load`, at the test's first `tensor()` of round 0.
- The existing store test ("released expert planes keep every shard of a load open…") passes unchanged.

## Branch 2: `flashnext-gather-overlaps-pack` (`7e3b1b726`, on branch 1)

### Design

1. Two `direct_gather.Gather` arenas. Chunk 0 is gathered into arena 0 before the chunk loop, as before.
2. In the loop, chunk k + 1's planes are gathered into arena (k + 1) mod 2, on a thread of their own (`Next`, with the gather's own 4 reader threads). Meanwhile chunk k packs from arena k mod 2 on the pack threads and uploads. The thread is joined before the next iteration, and also on any error path, through a `defer`. If a thread cannot start, the gather runs inline, as `pack` does with its threads.
3. While a gather runs, the main thread allocates nothing from the load's allocator (`pack` uses the page allocator). So the allocator is never used by two threads at once, and the order of allocations is the same as before.
4. The packed words and their order are the same: each chunk is packed from the same bytes into the same staging and uploaded to the same offset. Only when the next chunk's bytes are read changes.
5. Unchanged:
   - `TF_LOAD_DIRECT=0`, the mapped path: no second arena, no thread;
   - the staging and its growth count;
   - every free on failure: the gather thread is joined before anything it reads into is freed.

### The extra memory

- **The bound:** the second arena holds one chunk's planes, so a load needs at most one chunk's gather extra. It is allocated once, at chunk 1's gather, and reused after that.
- **At Flash-Next's geometry** (`dims` 2560, expert width 640, 512 experts, 192 MiB staging):
  - packed bytes per expert: 1,843,200 (gate and up) + 921,600 (down) = 2,764,800, so a chunk is 72 experts and a layer is 8 chunks (7 × 72 + 8);
  - one chunk's planes: 72 × 3 × (819,200 code + 102,400 E4M3 scale bytes) = 199,065,600 bytes (about 190 MiB). CSF scales make it smaller;
  - the gather lays out joined ranges from 4 KiB-aligned starts and reads gaps of up to 1 MiB between ranges in the same file. With each expert's three projections in their own tensors, that adds at most about 1 MiB per joined range.
  - **So the extra is about 190–200 MiB** of host memory, while experts load.
- Layers with fewer experts than one chunk use no second arena.

### Tests and planted controls

- **"mapped or direct: the same words"** (the 36-shard test, chunks of (5, 5, 2) and one whole-layer chunk, plain and CSF layers): passes unchanged. It also checks that the staging grew once.
- **New: "a direct load gathers each next chunk while the chunk before it packs, with the same words":**
  - A test store adds a hook (`chunkMark`) that the loader calls at each chunk's gather start and pack end. Only a store that declares it is called, so the weight store is unchanged.
  - The hook is a rendezvous: chunk k + 1's gather start waits until chunk k's pack has ended, and chunk k's pack end waits until chunk k + 1's gather has started, each for 5 s at most.
  - Only the two running at the same time can meet. With chunks (5, 5, 2), both sides meet twice (4 meetings), and the words are the loop's words.
  - **Now:** OK. All 173 tests in its binary run in 2.8 s, so no wait timed out.
  - **Planted control A** (chunk k + 1 gathered on its thread, but started after chunk k's pack): `expected 4, found 2`.
  - **Planted control B** (chunk k + 1 gathered inline before chunk k's pack, no thread): `expected 4, found 2`.
  - In both, the same words are packed. Only the side that waits for an event that already happened meets.
- **Allocation-failure walk, extended:** "a host allocation failing at any point…" now runs each mode with one whole-layer chunk and with chunks (5, 5, 2). Every failure point returns `OutOfMemory`, leaves no device memory, and the next load works.

  | mode | failure points, one chunk | failure points, 3 chunks |
  |---|---|---|
  | mapped | 3 | 3 |
  | direct | 14 | 23 (adds chunks 1 and 2: their gathers on the side thread, into the second arena) |

  The test now also asserts that the chunked direct walk is longer than the one-chunk walk.
- The other loader tests pass unchanged: the threads, the staging, the device-failure walk and two stores.

## Gates

On each branch head:
- **Clean:** `zig fmt --check` on the touched files, `git diff --check`, and `lean_check.py` + `lean_new.py` (424 on the base `6eff834ab`, 424 on each head, **0 new**).
  - A first branch 2 commit had one two-line `//` comment in the test. I made it one line and amended before pushing; the build gates were run again on the amended head.
- **Under 600 lines:** `cuda_prompt_wt.zig` 588 (578 before), `cuda_prompt_wt_test.zig` 144, `cuda_modelopt_threads.zig` 306, `cuda_modelopt_threads_test.zig` 585. I did not split anything.
- **Commits:** author and committer Jürgen Schmied, no `Co-Authored-By` or other trailer.

| branch | steps run | result |
|---|---|---|
| 1 | `zig build test`; `zig build test-cuda-host` (no `-Dnvcc`) | 462/465 and 283/285 steps; the 3 known failures; the Flash-Next host binary 172/172 |
| 2 | `zig build test test-cuda-host` (no `-Dnvcc`) | 462/465 steps; the 3 known failures; the Flash-Next host binary 173/173 |

**The 3 known failures**, all in `test-cuda-host`, in code neither branch touches:
- `listener`: accepted IPv4 and IPv6 TCP streams disable Nagle for incremental responses;
- `drama_voice_tool`: cleanup restores private directory permissions after encoder exit;
- `drama_voice_tool`: cleanup failure warns after publication and accompanies encoder failure.

They are sandbox failures. Upstream main `3ce4a0480` fails the same three here. I did not run `test-cuda-host` on `6eff834ab` itself.

## What I could not build or run

- **No nvcc and no GPU here.** `zig build test-cuda-host` was run without `-Dnvcc`: it builds the host-side Flash-Next code and runs its host tests, with the fake driver of the loader tests. No CUDA kernel was compiled, and nothing ran on a device.
- **No real checkpoint, no GB10.** The device bytes and replies on the real model are for your cold-load ABBA. The host tests show the same packed words on 36-shard plain and CSF fixtures.
- **No timing here.** The gains below are estimates from #1901's numbers.

## Expected load-time gain

From #1901: 48.0 s cold load, about 22 s outside the experts, gather about 405 ms a layer at 3.2 GB/s, 116 `use()` opens taking 4.1 s.

- **Branch 1: 2.5–4 s.**
  - The 116 opens cost about 35 ms each (open, mmap, header parse).
  - After the change, a shard is opened at most once per load by `use()`, and not at all if `hold()` already opened it (the expert loader keeps its shards in `released` until `loaded()`).
  - If `use()` still has to open S shards of its own, the saving is (116 − S) × 35 ms. With S between 0 and about 40 (the checkpoint's shard count bounds it), that is 2.7–4.1 s.
  - I take 2.5–4 s because parked shards now stay mapped, with a small cost in mappings.
- **Branch 2: 3–5.5 s.**
  - Assume about 26 s for the experts over 48 layers, about 540 ms a layer: 405 ms gather, and about 135 ms of pack, upload and globals.
  - With 8 chunks a layer, the pipeline hides the smaller of gather and pack in 7 of the 8 chunks. That is about 7/8 × 135 ms, or about 120 ms a layer, and about 5.7 s over 48 layers if the overlap were perfect.
  - The gather is disk-bound (O_DIRECT, little CPU), but the pack threads and the gather share memory bandwidth, and the upload stays serial. So I expect 55–95 % of that: 3–5.5 s.
  - If packing is in fact slower than gathering, the bound flips to about 7/8 of the gather time, and the gain is larger.
- **Together: about 5.5–9.5 s**, a 48 s load to about 39–42 s.
- **How to check it:** your phase timers should show the 116 opens gone from the non-expert path, and an expert layer close to max(gather, pack) instead of their sum.
