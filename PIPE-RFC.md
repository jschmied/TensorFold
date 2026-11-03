# RFC: a layer-range (pipeline) layout across mixed nodes

Draft by Jürgen Schmied. Against upstream `main` `933b969fa`, and #1922 (Starlight) at its head `39ecd68b3`.

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
- the first stage also holds the embeddings, and the last also holds the final norm and the head;
- each stage keeps KV / recurrent state for its own layers only;
- the stages form a chain. The last stage samples, and sends the token id back to the first.

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
- **Speed against fit:** at one stream a pipeline is not faster than one node; each stage idles while the others work. What it buys is fit: a model larger than one node, or a GB10 and a Mac serving one model. Throughput needs micro-batching across slots (question 5).

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

## 6. Where it plugs in

**Engine** (per family; the real work):
- a stage mode, `--layers a..b` with `--next host:port` and `--prev`, that:
  - loads only its layers' tensors (plus the embeddings / head at the ends);
  - runs the forward over those layers on incoming hidden rows;
  - keeps KV / state for those layers;
  - forwards rows to the next stage. The last stage samples and returns the token.
- **Weights:** the cluster's loader already reads per-rank byte ranges (`cluster/loader.zig:1`, `cluster/metal_sink.zig:31-38`). A CUDA counterpart of the Metal sink is needed. Flash-Next's loader already reads per-layer prefixes (`zig/src/families/flashnext/cuda_modelopt_experts.zig:64`).
- **Order:** Qwen 27B (dense GDN) first. Then Flash-Next, whose HC streams and PLE token inputs make the stage interface wider **(guess: +3–4 days per backend)**.

**Cluster runtime:**
- `backend.Shard.forward` (`cluster/backend.zig:11-23`) gains a hidden-row input and output for a stage.
- A TCP link implements `transport.zig`'s interface, which today says "never TCP" (`cluster/transport.zig:1`). The config allows `"transport": "tcp"` for pipeline-only layouts (today MCDMA only, `cluster/config.zig:115-116`).
- The planner change is from section 3.
- This also unblocks a Linux cluster build. The module already compiles and passes its tests on Linux (`CLUSTER-AUDIT.md`, section 6).

**Starlight** (one more layout beside `tensor_parallel`):
- A deployment field `pipeline: <family>`, with `nodes: N`.
- The controller asks the planner for ranges, instead of splitting `memory_gib` evenly (`sl/docs/fleet.md:39`).
- Each member gets a `Group`, as tensor parallel does (`sl/internal/control/wire/ranks.go:39-46`): `Member` is the stage index, `FirstRank` / `World` its layer range and the model's layers, and `Master` / `Port` the next stage's address, from the node's `rank_host` (`sl/internal/control/sync.go:133-145`).
- As with rank groups, the first stage's node serves and the others serve nothing (`sl/docs/nodes.md:34-36`).
- Draining a pipeline drains all its stages together (section 4, b).

## 7. Steps

Each step is useful alone. Days are for one engineer who knows the code **(guess)**. The audit's 19–26 days (`CLUSTER-AUDIT.md`, section 7) are refined here for a pipeline only.

| # | step | useful alone because | days | needs |
|---|---|---|---|---|
| 1 | Hidden-row wire format and a TCP stage link (framing, back-pressure, timeouts), tested over localhost with the cluster's toy model in two stages: same tokens as one rank | the transport every later step uses | 2–3 | host |
| 2 | Qwen 27B CUDA stage mode (`--layers`, `--next`), two processes on one GB10 or two GB10s over Ethernet; same bits as one node | splits a model too large for one GB10, with no RoCE | 4–6 | GPU |
| 3 | Planner: split by time with a memory cap, width-1 stages on mixed backends, a Linux / CUDA probe and per-family calibration; `cluster check` prints the split | correct ranges for any mix, used by Starlight or the CLI | 2–3 | host |
| 4 | Qwen 27B Metal stage mode; two Macs over Ethernet or Thunderbolt IP | Mac pairs without MCDMA | 4–6 | Mac |
| 5 | Starlight `pipeline` layout: placement from step 3, stage groups, drain as one | fleet placement and draining of pipelines | 2–3 | host |
| 6 | Churn (b): stage loss fails in-flight requests (retryable), re-plans, and serves anew | survives a node leaving | 1–2 | host |
| 7 | GB10 + Mac end to end, with the exactness wording from section 5 checked (repeatable per placement, within tolerance of each single node) | the mixed goal | 2 | GPU + Mac |
| 8 | Flash-Next stage mode (HC streams, PLE inputs) on both backends | the second family | 6–8 | GPU + Mac |

- **Total:** 17–25 days for Qwen 27B (steps 1–7), and 23–33 days with Flash-Next.
- **Host-only:** steps 1, 3, 5 and 6 (7–11 days).
- **Compared with the audit:** the Linux cluster build, the cluster-side Shard work and the app profile are folded into steps 1, 3 and 5, and churn re-prefill (option a) is left out.

## 8. Open questions for the maintainer

1. **Where should the layout live?**
   - Engine flags placed by Starlight (simplest, like `tensor_parallel` env ranks);
   - the cluster runtime's rounds (exact batching, but MCDMA-shaped today);
   - or engine flags plus the cluster planner as a library? I'd propose the last.
2. **Hidden rows on the wire:** the engine's own dtype (exact, larger), or bf16 (half the bytes, different bits)? For Qwen 27B these are the same, since its own rows are bf16 (section 2); the question is for Flash-Next.
3. **First family:** Qwen 27B (simpler, dense GDN), or Flash-Next (the one we want to serve)?
4. **Is "deterministic per placement, within single-node tolerance" acceptable** as the promise for CUDA + Metal pipelines?
5. **Micro-batching:** v1 serves one round at a time through the chain, so stages idle while others work. Should v1 add micro-batches across slots, or keep it for later?
