# CLUSTER-AUDIT: what TensorFold's cluster runtime can do, and what a layer split across a GB10 and a Mac still needs

Read only, host only (Linux, no GPU), on upstream `main` at `ba7b2c403`. No code changed. Paths are under `zig/src/cluster/` unless given in full; line numbers are from that commit.

## Short answer

- **The planner can already place uneven layer ranges by each stage's GPU memory.** Within a stage it splits reductions by bandwidth. Layer ranges never follow speed, and every stage has the same number of ranks.
- **Planning and estimation exist for two families** (Kimi K3 / Kimi Linear, GLM-5.3), plus a generic path from a local checkpoint. **Nothing runs a real model's layers across pipeline stages:**
  - the only `Shard` is a toy;
  - the K3 program checks one tensor-parallel matmul and one expert-parallel layer on one Mac.
- **Today a GB10 cannot join.**
  - The cluster is built and tested only in the macOS build.
  - The data plane is MCDMA only, with no TCP.
  - Probes are macOS tools.
  - There is no CUDA shard.
  - `serve` and `node` exit with "this build does not link the MCDMA endpoints".
- **Churn:** membership tracks joins, leaves and leaders, but nothing consumes a view change. A lost node ends the round with a timeout. Nothing re-plans, and a lost stage's KV is gone.
- **Exactness:** the promise is the same bits as one node, through a fixed-order ("canon") reduction. Metal and CUDA in one tensor-parallel group are refused ("different bits"); separate pipeline stages per backend are allowed by the config check.
- **Tests on this Linux host:**
  - the 3 Linux cluster steps pass;
  - the cluster module's own suite (what macOS `test-cluster` runs, minus the Metal sink) passes here when run by hand: 76 pass, 6 skip, 0 fail;
  - the GLM canon test passes 5/5.

## 1. Planner

**Layouts.** The planner gives pipeline × tensor parallel, with expert parallel inside tensor parallel.
- `Layout{tensor, pipeline, expert}` (`plan.zig:16-20`).
- `Stage{layers: Range, first, tensor, expert, slices, groups}` (`plan.zig:54-63`) and `Plan` (`plan.zig:65-72`).
- Each tensor is placed as replicate / column / row / expert / single / skip (`plan.zig:49`).
- Modes are `converged` and `disaggregated` (`plan.zig:13`; `cluster.schema.json:97-99`).
- Defaults: with no degrees given, there is one stage with tensor parallel over every node (`plan.zig:117-122`).
- `pipeline × tensor` must equal the node count, or `BadLayout` (`plan.zig:123`). `expert` must divide `tensor` (`plan.zig:124-125`). So every stage has the same number of ranks.

**Uneven layer ranges: yes, by memory.**
- **Layer ranges, by memory:** a stage's weight is the sum of its ranks' `gpu_limit` in GiB (`plan.zig:153-157`). `splitLayers` cuts contiguous layer ranges greedily at each stage's weighted share of the total layer bytes (`plan.zig:159`, `plan.zig:177-196`). So a 128 GiB node gets about twice the layers of a 64 GiB node.
- **Within a stage, by speed:** reduction slices are split by `weight(inv)`, which is bandwidth, else `gpu_limit` (`plan.zig:141-144`, `plan.zig:162-164`, through `canon.partition`). Expert groups use summed bandwidth (`plan.zig:166-171`).
- **Not done:**
  - layer ranges by speed or FLOPs;
  - any choice of node order: stage i is nodes `[i·tensor, (i+1)·tensor)` in id order (`plan.zig:156, 161`);
  - stages of different widths.
- **Tests:** every plan, cost and admission test uses identical 512 GiB Studios (for example `plan_test.zig:57`, `cost_test.zig:23`). The one exception lowers one node's wired limit (`plan_test.zig:109`). No test checks an uneven memory split across stages.

**Inputs.**
- The entry point is `plan(a, Checkpoint, *Shape, []Inventory, Options)` (`plan.zig:130`).
- `Options` (`plan.zig:27-47`) holds the mode, the layout, `slices`/`groups` (8 each), context, rows, chunk, the MLA split, drafter bytes, vision, margin and the OS floor.
- `Inventory` (`node.zig:46-65`) holds backend (`metal`/`cuda`, `node.zig:7`), memory, `gpu_limit`, bandwidth, GPU cores and `unified`.
- Heads must divide by slices and experts by groups (`plan.zig:131-132`).

**Who calls it.**
- Production: `cli_model.planModel` (`cli_model.zig:83`). It tries expert ∈ {0, 1} and the MLA splits (`cli_model.zig:75-89`) and keeps the cheapest admissible plan (`admission.best`, `cli_model.zig:93`).
- `planModel` is called by `cluster check` and `serve` (`cli.zig:167`, `cli.zig:208`), with options from `config.Model.options()` (`config.zig:60-66`).
- Inventories come from the config and probes, not from membership views (see 4).

**Mixed nodes in budget and cost.**
- `budget.zig` checks each rank against its own `gpu_limit` (`budget.zig:121-123`, `budget.zig:137-148`). A wired-limit raise is offered only to unified-memory Metal nodes (`budget.zig:126`).
- `cost.zig` prices FLOPs by backend: Metal by cores, anything else a flat constant (`cost.zig:92-95`). A stage costs as much as its slowest rank (`cost.zig:137-149`).
- Pipeline hand-offs are priced (`traffic.zig:94-99`) but never run (see 2).

## 2. Models

**What can be planned and estimated.**
- **Shapes from a config file plus a file list:**
  - `glm_moe_dsa` → `estimate.glm` (`cli_model.zig:59`, `estimate.zig:198`);
  - `kimi_linear` → `estimate.kimi` (`cli_model.zig:60-63`, `estimate.zig:133`);
  - any other family → `EstimateUnsupported` (`cli_model.zig:62`).
  - There are literal shapes `model.k3()` (`model.zig:170`) and `model.glm53()` (`model.zig:206`).
- **From a local model folder:** any family plans through `checkpoint.load` (real headers) plus `model.fromConfig` (`cli_model.zig:65-67`). But `fromConfig` only knows Kimi's linear layers (`linear_attn_config.kda_layers`, `model.zig:121-129`). Every other layer counts as GQA or MLA (`model.zig:119-120`), so for Qwen's `layer_types` / `full_attention_interval` the KV and state estimate is wrong.
- **Tensor roles:** `roles.zig` classifies `linear_attn.*` and `mlp.shared_expert` (`roles.zig:64, 73`). Packed experts without an `experts.N` index are replicated (`plan.zig:212`). PLE / n-gram tables fall to `.other`, so they are replicated.

**What runs: no forward over a stage.**
- **The interface:** `backend.Shard` is a vtable `forward / keep / release / vocab` (`backend.zig:11-23`). Its comment names "Metal, CUDA or the toy", but the only implementation is the toy's (`toy_cluster.zig:21-25`).
- **The toy** runs every layer on every rank (`toy.zig:153`). It is tensor / expert parallel only, with no stages.
- **Other pieces:**
  - `round.zig` only encodes the leader's row commands (`round.zig:17-89`);
  - `exchange.zig` has collectives only, with no stage-to-stage send (`exchange.zig:116-224`);
  - `metal_sink.zig` only wraps file ranges in Metal buffers (`metal_sink.zig:1, 31-38`);
  - `bringup.zig` is a simulated agreement (`bringup.zig:20`), and `serve` says "rounds need the real endpoints and kernels" (`cli.zig:213`).
- **`tf-k3-cluster`** (`zig/tests/kimi_k3/cluster.zig`; built at `zig/build/metal_targets.zig:437-445`) runs on one Mac with synthetic weights. It checks:
  - TP 2/4/8 on one `o_proj`, all-reduced in memory (`cluster.zig:51-100`);
  - EP 2/4/8 on layer 1's routed experts (`cluster.zig:114-164`).

  Both are bit-equal to one node. It runs no layers end to end and no pipeline.
- `families/kimi_k3/parallel.zig:19-20, 60` has `first_layer` / `last_layer`, but nothing calls them.

**What Flash-Next or Qwen 27B would need here.**
1. **`model.fromConfig`:** Qwen's `layer_types` / `full_attention_interval`, and the GDN key and value heads, head dims and conv kernel. Flash-Next's own reader is `families/flashnext/config.zig:225-247`.
2. **`Shape` fields:** separate linear K/V heads, `shared_expert_intermediate_size` (`shared` is a count, `model.zig:22`), PLE / n-gram, and the sparse indexer. `statePerStream` assumes KDA geometry (`model.zig:77-81`).
3. **Optional, for planning without the weights:** an estimator per family like `estimate.kimi` / `estimate.glm` (`estimate.zig:76-198`), dispatched in `cli_model.zig:59-63`.
4. **`roles.zig`:** classes for PLE / n-gram (replicate or stage 0) and Flash-Next's packed expert names.

## 3. Backends and transport

- **What runs a shard:**
  - On the Metal side, only a weight loader (`metal_sink.zig`) and the toy.
  - CUDA is a config value only: `Backend{metal, cuda}` (`node.zig:7`), `--cuda` (`cli.zig:111-113`), config parsing (`config.zig:152-153`).
  - I found no CUDA code path in `zig/src/cluster/`; I searched for `cuda`, `linux` and `builtin.os`.
- **Transports:**
  - The control plane says "never TCP" (`transport.zig:1`).
  - The real links are `fabric_link.zig`: one-sided RDMA windows (`fabric_link.zig:37`) and Thunderbolt two-sided send/recv (`fabric_link.zig:179`), which always reports `.tb5` at 80 Gb/s (`fabric_link.zig:245`).
  - The config refuses anything but MCDMA: "the data plane is MCDMA direct memory access only (no TCP)" (`config.zig:115-116`; `cluster.schema.json:12`).
  - Link kinds are `tb5 | cx5 | cx7` (`node.zig:10`).
  - Weight pulls go over the same links (`pull.zig:1, 100-120`).
  - `fake_net.zig` is in memory ("no sockets", `fake_net.zig:1`).
- **A separate CUDA fabric exists:**
  - `zig/src/fabric/tp/` covers NCCL, RoCE, and TCP for rank bootstrap, settings agreement, the round plan and fail-fast (`fabric/tp/bootstrap.zig:1`, `planlink.zig:1`, `fate.zig:1`).
  - The CUDA engines use it (`zig/build/cuda_engines.zig:13`); the cluster does not.
- **Could a GB10 over Ethernet join today? No.** In order:
  1. **No cluster build on Linux.**
     - The cluster module, `tf-cluster`, `test-cluster` and `metal_sink` are wired only in `metalTargets` (`zig/build/metal_targets.zig:237-245, 411-426, 454-456`), which only macOS calls.
     - The Linux CLI (`zig/src/cli/cuda_main.zig`) has no cluster command.
     - The module does compile on Linux, with the Metal engine module (`zig/src/tensorfold.zig`) as its `core`: I ran its tests here (see 6).
  2. **`serve` and `node` refuse:** "this build does not link the MCDMA endpoints… nothing runs over TCP", exit 3 (`cli.zig:202-204`), and "joining needs the MCDMA endpoints" (`cli.zig:226-227`).
  3. **Probes are macOS commands** (`sysctl`, `system_profiler`, `df /System/Volumes/Data`; `launch.zig:76-83`), parsed as macOS output (`probe.zig:158`). The node's memory limit follows macOS defaults (`node.zig:83-92`).
  4. **No Ethernet link kind.** Topology edges come from Thunderbolt UUIDs (`topology.zig:23-27`, `probe.zig:186-190`).
  5. **No TCP transport** (see above), and a Mac cannot join a RoCE fabric anyway.
  6. **No CUDA shard and no stage forward** (see 2).
- **The app** (`app/electron/cluster-node/`):
  - It does Bonjour / Avahi discovery (`advertise.ts:6, 23-29`) and an HTTPS management API (`README.md:13-18`).
  - It reads Linux links passively as unavailable (`passive-links.ts:34-36`).
  - It starts only a single-process engine. The only profile is `single-native` with `ranks: 1` (`native-adapter.ts:69`), and anything else is refused (`native-adapter.ts:87`).
  - `app/docs/clustering.md:9`: "The released engine's generic cluster commands do not launch a distributed model. The app must refuse those paths, including heterogeneous Metal/CUDA execution…" Line 35 says the same about mixed Mac/CUDA weight distribution.

## 4. Membership and churn

- **Membership:**
  - States are alive / suspect / dead / left (`membership.zig:12`).
  - A silent node turns suspect, then dead after 3 s (`membership.zig:18, 304-315`). A `.leave` drops it at once (`membership.zig:172-175, 399-405`).
  - `recompute` makes the lowest id leader (`membership.zig:335-338`). The leader bumps the epoch when the view digest changes, and followers adopt it (`membership.zig:340-352`). Each change is logged as an `Event` (`membership.zig:68, 119-120`).
- **Nothing consumes a view change in production.**
  - The epoch, events and `setPlan` / `agreed` are read only by:
    - the tests (`membership_test.zig:53-88`);
    - `status` printing the epoch (`status.zig:83`);
    - the one-time simulated bring-up (`bringup.zig:33-44`, reached only with fake probes, `cli.zig:209`).
  - The planner takes the config's inventories (`cli_model.zig:83`), and rounds never look at membership (`backend.zig:32-47, 113-125`).
- **A node that leaves mid-run stops the run, with no re-plan.**
  - The exchange or the round barrier times out with `PeerTimeout` (`exchange.zig:39, 107`; `barrier.zig:51`; mapped in `fabric_link.zig:98, 213`).
  - The error leaves `forward` / `follow`, and the toy follower panics (`toy_cluster.zig:93`).
  - `barrier.late()` is meant for suspicion but has no caller outside its test (`barrier.zig:67-76`).
  - A node that joins changes only the view and the epoch.
- **KV and re-placement: not found.**
  - KV is per slot on each rank and changes only through `keep` / `release` (`round.zig:12-13`, `backend.zig:49-52`). Nothing re-sends history to rebuild a lost stage, so its KV is lost.
  - Nothing re-places layers.
  - I searched for `re-?plan|re-?place|reshard|recompute|rebalance|migrat|failover|resume|re-?prefill`. The only hits are membership's view `recompute` and test crash helpers.

## 5. Exactness

- **The promise is the same bits as one node:**
  - "exact reductions: a fixed binary tree over S slices, with any contiguous split of the slices over ranks" (`canon.zig:1`);
  - "same bits on every rank" (`exchange.zig:179`);
  - the same tokens at 1, 2 and 4 nodes (`backend_test.zig:1, 111`), and a row's logits the same bits alone or beside others and at 1 or 4 nodes (`backend_test.zig:160`);
  - K3 TP / EP the same bits as one node (`zig/tests/kimi_k3/cluster.zig:1, 98, 162`).
- **It is shown on the toy and the K3 program only, never on a real model end to end.** The app docs say "no simulation can establish … distributed correctness" (`app/docs/clustering.md:27`).
- **What it refuses:**
  - **Metal and CUDA in one tensor-parallel group:** "Metal and CUDA kernels give different bits, so put each backend in its own pipeline stage or use "mode": "disaggregated"" (`config.zig:288-299`, tested at `config_test.zig:94`).
  - **Any transport but MCDMA** (`config.zig:116`).
  - **A foreign wire version** (`wire.zig:6, 95`).
  - **Weight mismatches:** `MissingShard` / `HashMismatch` / `SizeMismatch` (`loader.zig:89`).
  - **Layouts that don't fit:** `BadLayout` / `BadSlices` / `BadGroups` / `TooManyStages` (`plan.zig:114-132`), and a rank that does not fit (`budget.zig:137-148`, `admission.zig:19-20`).
- **Not compared between nodes:** OS version (probed at `probe.zig:30`, never compared), chip, GPU model, dtype.
- **Mixed backends in separate stages:**
  - The config allows it (`config_test.zig:95` accepts 2 Metal + 2 CUDA as TP 2 × PP 2).
  - One Mac plus one CUDA node on the default layout (one stage, tensor parallel over both) is refused with "different bits" (`config_test.zig:94`), so a GB10 + Mac model must say `"parallel": {"pipeline": 2}`.
  - But a reply then cannot be bit-equal to any one node: each stage's bits come from its own backend.
  - The promise would have to become the same bits as the same split run again (deterministic), not as one node. Nothing states or tests that yet.

## 6. Tests

**Test files (number of `test "` blocks):**
- `admission_test` 4, `backend_test` 7, `canon_glm_test` 2, `cli_test` 3, `config_test` 3, `cost_test` 3, `fabric_link_test` 3, `launch_test` 3, `loader_test` 3, `membership_test` 6, `plan_test` 3, `pull_test` 1, `fake_net` 1.
- Inline:
  - `barrier` 2, `bringup` 1, `canon` 3, `checkpoint` 3, `cost` 1, `estimate` 3, `exchange` 1, `metal_sink` 1;
  - `model` 3, `node` 2, `probe` 3, `roles` 1, `round` 1, `status` 1, `topology` 2, `traffic` 1, `wire` 2.
- `toy.zig` / `toy_cluster.zig` hold the toy model and the in-process clusters that `backend_test` and others drive.
- `cluster.zig:35-39` runs `refAllDecls` and imports the toys.

**What they cover:**
- exactness at 1–4 nodes (plain, drafted, shared, split experts) and NaN / Inf draw refusal (`backend_test`);
- membership joins, a leader crash, leave, a cut cable, gossip and refutation (`membership_test`, on `fake_net`);
- exact sums on fake fabric pipes and Thunderbolt two-sided links (`fabric_link_test`);
- plan, cost and admission for K3 and GLM on identical Studios;
- config errors;
- `ssh` launch through a jump host;
- weight reads, zero-copy maps and peer pulls checked chunk by chunk (`loader_test`, `pull_test`).

**Steps:**

| step | where | what runs |
|---|---|---|
| `test-cluster` | macOS only (`zig/build/metal_targets.zig:454-456`; not part of `test`) | the cluster module (all of the above but `canon_glm_test`) and `metal_sink` |
| `test-cluster-checkpoint-host` | any host (`zig/build/safetensors_header_host.zig:10-11`) | `checkpoint.zig` |
| `test-cluster-loader-host` | any host (`zig/build/safetensors_header_host.zig:13-14`) | `loader_test.zig` and what it imports |
| `compile-cluster-header-host` | any host | a compile only |
| `test-cuda-host` (in `test`) | any host | `canon_glm_test.zig` (`zig/build/cuda.zig:486-487`) and `checkpoint.zig` |

`tf-k3-cluster`, `tf-k3-validate` and `tf-tp2-bench` are macOS programs, not tests. They need a Metal GPU, a K3 checkpoint, and two Macs over MCDMA, in that order.

**Results here (Linux x86-64, Zig 0.17.0, `ba7b2c403`):**

| run | result |
|---|---|
| `zig build test-cluster-checkpoint-host` | 4/4 pass |
| `zig build test-cluster-loader-host` | 21 pass, 2 skip (`model` "the real K3 config (TF_K3_DIR)…", `estimate` "the real K3 file list (TF_K3_DIR)…") |
| `zig build compile-cluster-header-host` | success |
| the cluster module, run by hand with `test-cluster`'s imports (`lanes`, `fabric`, `core` = `zig/src/tensorfold.zig`, and `sys`) | **76 pass, 6 skip, 0 fail** (82). The skips: `topology` needs `TF_CLUSTER_PROBES`; `model`, `estimate` and `cli_test` need `TF_K3_DIR`; `admission_test` and `cost_test` print report tables only with `TF_CLUSTER_REPORT=1` |
| `canon_glm_test.zig` by hand (part of `test-cuda-host`) | 5/5 pass |
| `metal_sink` test, `tf-k3-*`, `tf-tp2-bench` | not run: they need macOS and Metal |

## 7. The gap: a GB10 plus one Mac running one model split by layers

**Target:** two pipeline stages, one per node, each on its own backend (the config already allows that, `config.zig:298`). The two nodes are linked over Ethernet with TCP, and the model is one that both engines already run (Qwen 27B or Flash-Next). Days are for one engineer who knows the code. "Host" means it can be built and tested without a GPU.

1. **Linux cluster build. 1 day, host.**
   - Wire the cluster module and `tf-cluster` (or `tensorfold cluster` in the CUDA CLI) into the Linux build, and run `test-cluster` there.
   - The module already passes its tests here (6). What is missing is the build wiring (`zig/build/metal_targets.zig:237-245`), and a decision whether its `core` stays the Metal engine module (which compiled here) or becomes the CUDA side's core.
2. **A TCP transport for the control plane and hand-offs. 2–3 days, host.**
   - Implement `transport.zig`'s link interface over TCP, which gives membership and pulls.
   - Allow `"transport": "tcp"` for pipeline-only layouts (`config.zig:116`, `cluster.schema.json:12`).
   - Add an `eth` link kind (`node.zig:10`) and a TCP edge in topology.
   - `fabric/tp/bootstrap.zig` and `planlink.zig` hold reusable TCP pieces. Test it on `fake_net`'s model and over localhost.
3. **A Linux / CUDA probe. 1 day, host.**
   - Read memory, GPU and unified memory from `/proc/meminfo` and `nvidia-smi` instead of `sysctl` / `system_profiler` (`launch.zig:76-83`, `probe.zig:158`), with fixtures like `probe_fixtures.zig`.
   - Set the GB10's `gpu_limit` (unified memory) explicitly, not from macOS defaults (`node.zig:83-92`).
4. **The model's shape in the cluster. 1–2 days, host.**
   - Teach `model.fromConfig` Qwen's layer types and GDN / MoE fields, and teach `roles` the PLE / n-gram tables (see 2).
   - Then `cluster check` gives a correct layer split and budget from the real checkpoint (through `checkpoint.load`, no estimator needed).
5. **A stage hand-off in the round. 3–4 days, host.**
   - `Shard.forward` today takes rows and collectives only. A stage also needs its first and last layer, a hidden-state input from the stage before, and an output to the next. The last stage returns logits to the leader.
   - Fix the hand-off format (bf16 hidden rows, row order as in `round.zig`).
   - Add the pipeline to the toy (`toy.zig`, `toy_cluster.zig`) and test that two toy stages give the same tokens as one rank.
6. **The exactness rule for mixed stages. 0.5 day, host and docs.**
   - Define the promise for a mixed split: deterministic per placement, not equal to one node (see 5).
   - Refuse a re-placement that would move a stage to the other backend in the middle of a conversation.
7. **A CUDA stage shard. 4–6 days, GPU.**
   - The family's CUDA engine loads only layers [a, b), plus the embedding on stage 0 and the head on the last stage. It runs `Shard.forward` on hidden rows and keeps its own KV per slot (`keep` / `release`).
   - Weights come through `loader.zig`'s ranges into CUDA memory (a CUDA counterpart of `metal_sink.zig`).
8. **A Metal stage shard. 4–6 days, Mac.** The same on the Metal engine for that family, with `metal_sink.zig` as the weight path.
9. **End-to-end check and the app profile. 2 days, GPU and Mac.**
   - Run `serve` on the two nodes over TCP, with the prompt split across stages. Check the reply against the same split run twice (deterministic), and against each single node within the family's tolerance.
   - Add an app profile beside `single-native` (`native-adapter.ts:69, 87`).

**Total:** about 19–26 days. Steps 1–6 (about 9–12 days) are host-only and can be done and tested without either machine.

**Left out of this first cut:**
- **Churn handling:** today a lost node ends the run, which is acceptable at first. Re-planning on a view change and re-prefilling a lost stage's KV is about 3–5 days more.
- **Placing layers by speed:** `cost.zig` already prices stages, so splitting by time instead of memory is about 1 day.
