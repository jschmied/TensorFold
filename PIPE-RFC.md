# RFC: a layer-range (pipeline) layout across mixed nodes

Draft by Jürgen Schmied. Against upstream `main` `933b969fa`, and #1922 (Starlight) at its head `39ecd68b3`; section 6 against `main` `071cefe08`.

**Conventions:**
- `cluster/…` is `zig/src/cluster/…` and `sl/…` is `starlight/…` in #1922.
- Code claims are cited by file:line. Everything marked **(guess)** is an estimate, not measured.
- The cluster evidence is from `CLUSTER-AUDIT.md` (fork branch `cluster-audit-report`).

## 1. Problem

- **Tensor parallel needs fast links and the same bits on every rank.**
  - Every layer all-reduces partial sums, so one decode step makes dozens of round trips.
  - The cluster refuses Metal and CUDA in one tensor-parallel group: "Metal and CUDA kernels give different bits" (`cluster/config.zig:288-299`).
  - Starlight's `tensor_parallel` is CUDA (NCCL) or ROCm only (`sl/internal/control/wire/ranks.go:13-34`; `sl/docs/fleet.md:105-111`).
- **So a GB10 and a Mac, or two nodes on 1/10 GbE, can't share a model today.**
- **A layer split instead sends one hidden row per token per stage boundary.** That fits Ethernet, and each stage runs its own backend.

## 2. The layout

**A stage** is one node (or one tensor-parallel group of identical nodes) holding layers `a..b` of one model:
- rank 0 serves: it holds the lane host, the embeddings and the first layer range, and also the final norm, the head, the sampler and any drafter;
- the other ranks hold their layer ranges only, and each stage keeps KV / recurrent state for its own layers;
- the rows go round a ring: rank 0, 1, ..., the last, and back to rank 0 for the head. That costs one more hidden hop (20 KiB a row for Qwen 27B) than a chain, and keeps sampling, penalties, stop strings, the prompt cache's host side and MTP on one node, unchanged.

**What crosses the wire each step:**
- the hidden rows after layer `b`, in the dtype the engine itself keeps between layers. A narrower wire type would change bits (see 5);
- the token ids, for families whose layers read them (Flash-Next's PLE n-gram embeddings, **(guess)** that each stage needs them);
- the round's row map (slots, positions), as the cluster's round commands carry today (`cluster/round.zig:17-89`).

**Sizes per stage boundary.** Qwen 27B is `hidden_size` 5120 and 64 layers (`zig/src/families/qwen27/gdn_test.zig:9`). Between layers its CUDA engine keeps two bf16 rows, the hidden state and the residual (`e.h` and `e.r`), so its own dtype is already bf16 and an exact boundary carries 2 × 5,120 bf16 values. Flash-Next carries 4 residual streams of 2560 (`zig/src/families/flashnext/cuda_prompt_wt.zig:10-15`, `wide = streams * dims`).

| model | values a row | decode, 1 row | prompt chunk, 4096 rows |
|---|---|---|---|
| Qwen 27B (h and r, bf16, exact) | 2 × 5,120 | 20 KiB | 80 MiB |
| Flash-Next (f32 / bf16) | 10,240 | 40 KiB / 20 KiB | 160 MiB / 80 MiB |

**Latency budget per boundary.** I assume about 115 MB/s of payload at 1 GbE and about 1.15 GB/s at 10 GbE, with a LAN one-way latency of about 0.1 ms **(guess)**.

| | 1 GbE | 10 GbE |
|---|---|---|
| decode, Qwen 27B row (h and r) | ≈ 0.3 ms | ≈ 0.12 ms |
| decode, Flash-Next row (f32) | ≈ 0.45 ms | ≈ 0.14 ms |
| prompt chunk, Qwen 27B (h and r) | ≈ 0.73 s | ≈ 73 ms |
| prompt chunk, Flash-Next (f32 / bf16) | ≈ 1.46 s / 0.73 s | ≈ 146 ms / 73 ms |

- **Decode:** a token costs one hidden hop per boundary plus a 4-byte token back. A plain Qwen 27B (MLX 4-bit) decode step is about 64 ms on a GB10 (measured, section 2a), so a 0.3 ms hop at 1 GbE is about 0.5 %.
- **Prompts:** the chunks pipeline. Chunk k+1 computes on stage 1 while chunk k crosses the link and computes on stage 2. So the link only costs time when it is slower than a stage.
  - Qwen 27B (measured): a cold 32k prompt takes 25.0 s on one GB10, about 3.1 s a 4096-row chunk, so about 1.6 s a chunk on each of two stages. The 0.73 s a chunk at 1 GbE hides behind that.
  - At 1 GbE a slower link than a stage is likely for Flash-Next, whose MoE compute per row is small **(guess)**.
  - 10 GbE or more is the comfortable target for long prompts.
- **Speed against fit:**
  - **Decode** at one stream is not faster than one node: a token still passes every weight in order, and decode is bound by reading them. Overlapped micro-batches keep both nodes busy, but in the memory-bound regime that buys room for more streams (twice the KV), not faster streams.
  - **Prompts** can be faster at one stream. With chunks overlapped, a prompt of n chunks on two stages takes about (n + 1) / 2 chunk-times of one node's, so about 1.8x faster to the first token at 8 chunks (a 32k prompt), as long as the link is faster than a stage.
  - What a pipeline always buys is fit: a model larger than one node, or a GB10 and a Mac serving one model.

## 2a. A prototype: Qwen 27B in two stages on one GB10

To check the boundary before proposing it, I split Qwen3.8-27B (MLX 4-bit) into layers 0-32 and 32-64 in two processes on one GB10, with the boundary over localhost TCP (no Nagle). It is a test-tool mode, not a server, and both processes still load every weight.

- **Exact:** the last stage's logits equal one process's logits at all 128 greedy steps of a 2,000-token prompt, byte for byte, in both runs.
- **Decode:** 65.2 and 65.3 ms a step split, against 63.6 and 65.1 ms whole: up to 1.6 ms for two device copies and the hop, on a step that runs nothing in parallel.
- **Prompt:** one 2,000-row chunk takes 1,632-1,637 ms split against 1,536-1,543 ms whole. The 40 MiB of rows cost about 95 ms, all of it visible, because a single chunk leaves nothing to overlap.
- **One trap:** the CUDA engine's `init` sets its fields one by one, so a new field's default is never applied. A stage field left unset gave NaN logits on the whole path.

The next steps for the prototype are loading only each stage's layers, chunked prompts with overlap, and a second machine.

## 3. Planning

**What exists:**
- The cluster config already has a pipeline degree, `models.<id>.parallel.pipeline` (`cluster/config.zig:224, 285-289`), and its mixed-backend refusal already suggests it: "put each backend in its own pipeline stage" (`cluster/config.zig:298`).
- `plan.zig` already cuts uneven, contiguous layer ranges. Each stage's share is weighted by its ranks' `gpu_limit` (`cluster/plan.zig:153-159`), through `splitLayers` (`cluster/plan.zig:177-196`).
- Speed today only splits work within a stage (`cluster/plan.zig:141-144, 162-171`).
- Every stage has the same width (`cluster/plan.zig:123`), and stages take nodes in id order (`cluster/plan.zig:156, 161`).
- `budget.zig` checks each rank against its own `gpu_limit` (`cluster/budget.zig:121-123, 137-148`).
- `cost.zig` prices a stage at its slowest rank (`cluster/cost.zig:137-149`), with FLOPs per backend (`cluster/cost.zig:92-95`).

**Proposed:**
1. **Split by time, capped by memory.** For a pure pipeline (tensor = 1), give stage *i* layers in proportion to its measured speed for this family, then move layers off any stage whose weights + KV + state exceed its `gpu_limit` (budget's check). That is a change to `splitLayers`' weights only.
2. **Order stages** to keep the first and last stages on the nodes that hold the embeddings and head cheaply. Keep id order as the default.
3. **Allow width 1 per stage on mixed backends.** Keep the tensor-parallel same-backend rule (`cluster/config.zig:298`).

**What a node reports** (beyond the cluster's `Inventory`: backend, memory, `gpu_limit`, bandwidth, cores, unified, `cluster/node.zig:46-65`; and Starlight's `Facts`: platform, backend, memory, GPUs, `rank_host`, `sl/internal/control/wire/wire.go:56-68`):
- for each family it may serve, the measured ms per layer at decode (1 row, and B rows) and per 4096-row prompt chunk. One calibration run per family and engine version **(guess: about 30 s each)**;
- usable memory for weights and KV after the OS floor;
- for each link: the peer, the measured bandwidth and the round-trip time. A probe over the same TCP port the stages use.

## 4. Churn

**Today:**
- Membership tracks alive / suspect / dead / left (`cluster/membership.zig:12, 304-315`), but nothing reads a view change (`cluster/status.zig:83`, `cluster/bringup.zig:33-44`).
- A lost peer ends the round with `PeerTimeout` (`cluster/exchange.zig:39, 107`; `cluster/barrier.zig:51`).
- Starlight drains and stops any replica it no longer lists (`sl/docs/nodes.md:23-25`) and gives a node 180 s to drain on shutdown (`sl/docs/nodes.md:69`).

**Options when a node leaves mid-request:**
- **(a) Re-plan, then rebuild the lost stage's KV by re-prefilling from the request's tokens.** The server has the prompt and the tokens it generated.
- **(b) Fail the in-flight requests on that replica** with a retryable error, re-plan, and serve new requests on the new placement.

**Recommendation: (b) for the first version.**
- A re-plan can move layers to the other backend, so a resumed reply would change bits halfway (see 5).
- (a) also needs KV for every layer rebuilt in order, which is a full prefill anyway.
- (b) reuses Starlight's drain-and-restart path, and clients already retry 503s (`sl/docs/nodes.md:49`).
- (a) can follow as an option for same-backend re-plans, where bits are unchanged.

**Joins:** a new node changes nothing until the next planned re-placement. That happens at a drain boundary, never mid-request.

## 5. Exactness

**Today the cluster promises the same bits as one node:**
- through its fixed reduction tree (`cluster/canon.zig:1`; `cluster/exchange.zig:179`);
- tested at 1, 2 and 4 nodes on the toy model (`cluster/backend_test.zig:111, 160`).

**For a pipeline:**
- **Same backend, same kernels, hidden rows sent in the engine's own dtype:** the same bits as one node running the same kernels with the same row batches. A stage boundary moves data but does no arithmetic.
  - The condition is that each stage sees the same batch composition as one node would. Kernels that are not batch-invariant would otherwise differ **(guess: true for the current engines' row-batched paths; to be checked per family)**.
- **Mixed CUDA and Metal:** no promise of equality with either single node, because each stage's bits come from its own backend.
  - What we can promise: the same placement, engine versions and inputs give the same bits (deterministic per placement).
  - Each stage also stays within its family's single-node tolerance of a reference.
  - The served reply should say which placement produced it **(proposal)**.
- **Not promised:** bits across a re-placement, which is why section 4 recommends (b).

## 6. Fit with the current design

The CUDA engines already have a coordination layer: the tensor-parallel fabric (`zig/src/fabric/tp`). The pipeline should join it, not the cluster runtime (`zig/src/cluster`). The cluster runtime is MCDMA-only, builds only on macOS, has no CUDA shard, and its rounds never read membership (`CLUSTER-AUDIT.md`, sections 3 and 4).

### 6.1 Communication

**What exists in `fabric/tp`:**
- TCP rank bootstrap, a star on rank 0, and a settings agreement (`bootstrap.zig`).
- Rank 0's round plan to the followers over its own TCP connection (`planlink.zig`).
- Fail-fast: a failed or vanished rank ends all ranks (`fate.zig`).
- Point-to-point `send` / `recv` on the stream-ordered, graph-capturable `Collective` (`collective.zig:49-50`). Its NCCL backend serves two CUDA nodes, and the hop can then sit inside a captured decode graph.

**What is missing:**
- A TCP `Collective` backend implementing `send` / `recv` only, with no reductions. It is needed for plain Ethernet, for a Mac, and for two processes on one GPU, where NCCL refuses two ranks.
- GB10 and Apple silicon both have unified memory, so staging through the host is cheap.
- The Metal engine has no `fabric/tp` at all. It needs the TCP backend, bootstrap and the plan link on macOS.

### 6.2 Node coordination

- **Launch and roles** work as for `tensor_parallel` today. Starlight starts each rank with its environment (rank, master, port) and its layer range, and rank 0 serves.
- **Ranges** come from `cluster/plan.zig`'s `splitLayers` used as a library: by memory today, by measured speed later (section 3).
- **Churn** is fail-fast as it stands: one rank lost ends the replica, and Starlight restarts it. That is option (b) of section 4, so v1 needs no membership work.
- **Per-row metadata** (positions, slots, segments, draw rules) rides the plan link, as in tensor parallel.
- **New work on the followers**, the largest part of this layer:
  - each stage holds KV and recurrent state for its own layers a slot;
  - `keep` (the rollback to an accepted path after a verify) reaches every stage through the plan;
  - the prompt cache saves and restores each stage's part under one key;
  - rank 0's admission charges by the tightest stage. The followers report their pools at bootstrap, which fits the byte budget planned in #1943.

### 6.3 Scheduling

Today one lane host runs one round at a time on one engine thread:
- `step` lands any in-flight token before it plans (`core/lanes/engine.zig:128-131`);
- `verify` reads its tokens back before `step` returns (`families/qwen3_5/cuda_native.zig:416-474`, `cuda_engine.zig:544-546`);
- the engine has one CUDA stream, one set of activation buffers, one staged-table buffer and one graph map (`cuda_engine.zig:82-156`).

| level | what it is | fits today | what changes |
|---|---|---|---|
| v1: in turn | a round runs front, send, the other stages, receive, then the head | yes | `verify` and the prompt step become front, send, receive, head inside the family. The lane host is unchanged. The decode graph is split per stage, or holds NCCL's send and receive. |
| v2: prompt overlap | one prompt's chunks overlap across the stages | mostly | `prefill_step` issues chunk k + 1's front before chunk k's head returns, inside the family, with a second set of activation buffers. The lane host is unchanged. |
| v3: decode micro-batches | two groups of streams in flight | no | `verify` splits into issue and complete halves, with scratch and graphs for each group, and the lanes `Engine` plans two round sequences. That is the core lanes code every family shares. |

v3 gains room for streams and compute-bound batches, not per-stream speed (section 2), so it comes last.

**Any number of stages:** nothing here is specific to two. The ring runs rank 0, 1, ..., N - 1 and back to rank 0; the plan link is a star of up to 64 ranks (`planlink.max_ranks`); `splitLayers` cuts N ranges. With N stages a v1 token adds N hops (about 2 % at four stages on 1 GbE for Qwen 27B), each node is busy 1/N of the time, v2 approaches N times faster for long prompts, and v3 needs at least N groups.

**Drafters:**
- MTP taps only the last layer and reuses the head (`families/qwen3_5/cuda_mtp.zig:121, 157, 354`), so it stays on rank 0.
- DFlash2 taps the drafter's `target_layer_ids` (for example 5, 19, 33, 47 and 61) on whichever stage holds them (`cuda_forward.zig:116-118`). Those taps would ride back with the row, about 50 KiB more a row. v1 serves MTP or no drafter.

**Graphs:** decode is captured from embed to head today (`cuda_engine.zig:511-526`). Each stage needs its own capture, and on TCP the hop stays outside the captured region.

## 7. Steps

The test hardware is one GB10 and one Mac. That sets the order:
- two CUDA nodes over NCCL can't be tested, and NCCL refuses two ranks on one GPU, so the TCP backend comes first;
- the only two-machine test is GB10 + Mac, so a Metal follower comes before any speed claim;
- two processes on one GB10 share its GPU, so they show exactness, not overlap gains.

Days are for one engineer who knows the code **(guess)**.

| # | step | shown on | useful alone because | days |
|---|---|---|---|---|
| 1 | TCP `Collective` backend (`send` / `recv`, framing, deadlines) beside NCCL, with host tests and a localhost pair | host, GB10 | the link every later step uses; also Linux pairs without RoCE | 2-3 |
| 2 | Qwen 27B CUDA stage mode, v1: ring with the head on rank 0, any number of stages, follower KV, `keep` through the plan; two and three processes on one GB10 (three covers a middle stage) | GB10 | same bits as one node (as the prototype), serving through the lane host | 4-6 |
| 3 | `fabric/tp` on macOS: TCP backend, bootstrap, plan link, fail-fast | Mac | the Metal engine can join a rank group at all | 2-3 |
| 4 | Qwen 27B Metal follower stage | Mac | a Mac holds a layer range | 4-6 |
| 5 | GB10 + Mac over Ethernet, v1: deterministic per placement, each stage within its family's tolerance (section 5) | GB10 + Mac | the mixed goal, measured | 2 |
| 6 | v2: chunked prompt overlap | GB10 + Mac | the first measurable speed gain, time to first token | 2-3 |
| 7 | Planner by measured speed with a memory cap; Starlight `pipeline` layout | host | correct ranges for any mix, placed by the fleet | 3-4 |
| 8 | Prompt cache and admission across stages (with #1943's byte budget) | GB10 + Mac | resumed prompts and full admission on a pipeline | 3-4 |
| 9 | v3: decode micro-batches in the lanes core | GB10 + Mac | room for more streams | 5-8, after agreement |
| 10 | Flash-Next stage mode, and DFlash2 taps across stages | GB10 + Mac | the second family and the faster drafter | 6-8 |

- **Total:** 22-31 days to step 8, which is a served GB10 + Mac pipeline with prompt overlap, resume and admission. Steps 9 and 10 add 11-16 days.
- **Host-only:** steps 1 and 7 (5-7 days).
- **Not testable here:** two CUDA nodes over NCCL / RoCE. The NCCL `send` / `recv` path would need a second CUDA node, or ashhart's pair.

## 8. Open questions for the maintainer

1. **Where should the layout live?** I'd propose the `fabric/tp` rank model, with engine flags placed by Starlight and the cluster planner as a library (section 6), rather than the cluster runtime's MCDMA rounds. Is that the direction you want, or should the cluster runtime gain CUDA and TCP instead?
2. **Hidden rows on the wire:** the engine's own dtype (exact, larger), or bf16 (half the bytes, different bits)? For Qwen 27B these are the same, since its own rows are bf16 (section 2); the question is for Flash-Next.
3. **First family:** Qwen 27B (simpler, dense GDN), or Flash-Next (the one we want to serve)?
4. **Is "deterministic per placement, within single-node tolerance" acceptable** as the promise for CUDA + Metal pipelines?
5. **Micro-batching (v3):** it changes the lanes core every family shares (section 6.3). Is a later change there acceptable, or should pipelines stay at v1 and v2?
6. **A second CUDA node:** we have one GB10 and one Mac. Could the NCCL path (two CUDA nodes) be checked on your pair once steps 1 and 2 land?
