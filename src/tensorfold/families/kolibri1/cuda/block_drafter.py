"""Kolibri 1's block drafter (train_block.py's checkpoints): one pass drafts ``block`` tokens after the last kept row.

Each kept row r becomes a context row fc(cat(embed(token after r), fuse(tapped states at r))); every layer's
attention reads keys and values projected from those rows, so a stream keeps them per layer in a buffer and appends
new rows only. A draft pass at anchor t (the last kept row) runs the block: row 0 is t's own context row (its output
after layer 0 is the chain drafter's first guess), rows 1.. a mask embedding beside t's fused state; row j attends to
the context and to block rows 1..j. Position j's output, corrected by the token drafted at j-1 (the predecessor head),
is read by Kolibri's own head as the token at t+2+j. Drafts change no reply: the target verifies them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

WINDOW = 2048            # context rows an anchor sees (the training window)


def _quantize(w: torch.Tensor, mode: str):
    """A bf16 weight [N, K] as the lane matmul's projection: "fp8" (e4m3, an fp32 scale per row and 64 inputs) or
    "nvfp4" (e2m1, e4m3 scales per 16 inputs, one global scale, ModelOpt's recipe); "bf16" keeps the tensor."""

    if mode == "bf16":
        return w
    if mode == "fp8":
        from tensorfold.cuda.nvfp4.linear import Fp8BlockLinear

        n, k = w.shape
        g = w.float().view(n, k // 64, 64)
        top = g.abs().amax(-1)
        scale = torch.where(top > 0, top / 448.0, torch.ones_like(top))
        codes = (g / scale[..., None]).view(n, k).to(torch.float8_e4m3fn)
        return Fp8BlockLinear.from_rows(codes, scale.contiguous())
    if mode == "nvfp4":
        from tensorfold.cuda.nvfp4.experts import quantize
        from tensorfold.cuda.nvfp4.linear import Fp4Linear

        words, scales, glob = quantize(w[None])
        return Fp4Linear.from_checkpoint(words[0], scales[0].view(torch.float8_e4m3fn), float(glob[0]))
    raise ValueError(f"block drafter quantization {mode!r}: bf16, fp8 or nvfp4")


def is_block(path: str | Path) -> bool:
    """A train_block.py checkpoint (a .pt whose config has a block size), read without loading its tensors."""

    p = Path(path)
    if p.is_dir() or p.suffix != ".pt":
        return False
    return "block" in torch.load(p, map_location="cpu", weights_only=False, mmap=True)["config"]


class _Cache:
    """One stream's context: per layer keys and values of its rows, plus the last row's context and fused vectors."""

    def __init__(self, layers: int, h: int, hd: int, d: int, dev, dt, block: int) -> None:
        shape = (h, 2 * WINDOW + block, hd)                 # past the rows: the block's own keys, rewritten each pass
        # zeros: the pass attends over the whole buffer under a mask, and masked garbage could still be NaN
        self.k = [torch.zeros(shape, device=dev, dtype=dt) for _ in range(layers)]
        self.v = [torch.zeros(shape, device=dev, dtype=dt) for _ in range(layers)]
        self.n, self.pos, self.block, self.ready = 0, -1, block, False
        # the pass reads these device copies only, so a captured graph replays with the stream's current state
        self.n_t = torch.zeros((), dtype=torch.long, device=dev)
        self.pos_t = torch.zeros((), dtype=torch.long, device=dev)
        self.ctx = torch.zeros(1, d, device=dev, dtype=dt)
        self.fused = torch.zeros(1, d, device=dev, dtype=dt)
        self.rows = torch.arange(shape[1], device=dev)
        self.graphs: dict[int, tuple] = {}

    def room(self, rows: int) -> None:
        """Keep the last WINDOW rows at the front when ``rows`` more would not fit."""

        if self.n + rows + self.block > self.k[0].shape[1]:
            keep = min(self.n, WINDOW)
            for t in self.k + self.v:
                t[:, :keep] = t[:, self.n - keep : self.n].clone()
            self.n = keep


class BlockDrafter:
    def __init__(self, path: str | Path, embed: torch.Tensor, head: torch.Tensor, *, depth: int | None = None,
                 vocab: str | Path | None = None, dtype: torch.dtype = torch.bfloat16, quant: str | None = None,
                 head_quant: str | None = None) -> None:
        ck = torch.load(Path(path), map_location="cpu", weights_only=False)
        cfg = ck["config"]
        if not (cfg.get("chain_ctx") and cfg.get("row0_chain")):
            raise ValueError(f"block drafter {path}: serving reads chain_ctx + row0_chain checkpoints only")
        self.cfg, self.dt = cfg, dtype
        self.d, self.h, self.hd = cfg["hidden"], cfg["heads"], cfg["head_dim"]
        self.eps, self.theta, self.layers, self.block = cfg["eps"], cfg["theta"], cfg["layers"], cfg["block"]
        self.taps = tuple(int(t) for t in ck["taps"])
        if embed.shape[1] != self.d:
            raise ValueError(f"block drafter {path}: width {self.d} does not fit Kolibri's {embed.shape[1]}")
        dev = embed.device
        st = ck["state"]

        def w(name: str) -> torch.Tensor:
            return st[name].to(dev, dtype).contiguous()

        def n(name: str) -> torch.Tensor:
            return st[name].to(dev, torch.float32 if dtype != torch.float64 else dtype)

        self.fuse, self.fc, self.mask = w("fuse.weight"), w("fc.weight"), w("mask")
        self.L = []
        for i in range(self.layers):
            p = f"layers.{i}."
            L = {k: w(p + k + ".weight") for k in ("q", "k", "v", "ck", "cv", "o", "gate", "up", "down")}
            L.update({k: n(p + k + ".w") for k in ("n1", "n2", "nc")})
            self.L.append(L)
        self.out_norm = n("out_norm.w")
        self.spine = [(n(f"spine_n.{i}.w"), w(f"spine_a.{i}.weight"), w(f"spine_u.{i}.weight"))
                      for i in range(self.layers - 1)] if cfg.get("spine_rank") else []
        self.pred = (w("pred_h.weight"), w("pred_e.weight"), w("pred_out.weight")) if cfg.get("pred_rank") else None
        self.embed = embed
        ids = json.loads(Path(vocab).read_text()) if vocab else None
        self.vocab = torch.tensor(ids, device=dev) if ids is not None else None
        self.head = (head[self.vocab] if self.vocab is not None else head).to(dtype).contiguous()
        # the bytes a pass reads decide its cost on the GB10: weights and the head slice may be quantized (drafts
        # only: the target verifies every one)
        quant = quant or os.environ.get("TENSORFOLD_KOLIBRI_BLOCK_QUANT", "bf16")
        head_quant = head_quant or os.environ.get("TENSORFOLD_KOLIBRI_BLOCK_HEAD", quant)
        self.quant = (quant, head_quant)
        if quant != "bf16":
            for L in self.L:
                for k in ("q", "k", "v", "ck", "cv", "o", "gate", "up", "down"):
                    L[k] = _quantize(L[k], quant)
            self.fuse, self.fc = _quantize(self.fuse, quant), _quantize(self.fc, quant)
            self.spine = [(sn, _quantize(sa, quant), _quantize(su, quant)) for sn, sa, su in self.spine]
            if self.pred is not None:
                self.pred = tuple(_quantize(t, quant) for t in self.pred)
        self.head_mm = _quantize(self.head, head_quant)
        self.inv = 1.0 / self.theta ** (torch.arange(0, self.hd, 2, device=dev, dtype=torch.float32) / self.hd)
        self.depth = min(self.block, depth or self.block)
        self.cache: dict[int, _Cache] = {}
        self.states: list[torch.Tensor] = []               # the last pass's corrected states (tests read them)

    def drop(self, sid: int) -> None:
        self.cache.pop(sid, None)

    def _rms(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x32 = x.float() if x.dtype != torch.float64 else x
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps) * w).to(x.dtype)

    def _rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Neox halves over the last dim of x [H, T, Dh] at positions pos [T]."""

        ang = pos.float()[:, None] * self.inv[None]
        cos, sin = torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)
        half = self.hd // 2
        rot = torch.cat([-x[..., half:], x[..., :half]], -1)
        return (x.float() * cos + rot.float() * sin).to(x.dtype) if x.dtype != torch.float64 else x * cos + rot * sin

    @staticmethod
    def _mm(x: torch.Tensor, w) -> torch.Tensor:
        """x [M, K] times a projection: a bf16 tensor [N, K], or a quantized lane-matmul projection."""

        return x @ w.T if isinstance(w, torch.Tensor) else w(x.contiguous())

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], self.h, self.hd).transpose(0, 1)                     # [H, T, Dh]

    @torch.no_grad()
    def add(self, sid: int, positions: list[int], nxt: list[int], taps: list[torch.Tensor]) -> None:
        """Context rows for kept rows at ``positions`` (in order): the token after each and the rows' tapped states."""

        if not positions:
            return
        dev = self.embed.device
        c = self.cache.get(sid)
        if c is None:
            c = self.cache[sid] = _Cache(self.layers, self.h, self.hd, self.d, dev, self.dt, self.block)
        for lo in range(0, len(positions), WINDOW):
            part = slice(lo, lo + WINDOW)
            rows = len(positions[part])
            c.room(rows)
            pos = torch.tensor(positions[part], device=dev)
            fused = self._mm(torch.cat([t[part] for t in taps], -1).to(self.dt), self.fuse)
            ctx = self._mm(torch.cat([self.embed[torch.tensor(nxt[part], device=dev)].to(self.dt), fused], -1), self.fc)
            for i, L in enumerate(self.L):
                a = self._rms(ctx, L["nc"])
                c.k[i][:, c.n : c.n + rows] = self._rope(self._heads(self._mm(a, L["ck"])), pos)
                c.v[i][:, c.n : c.n + rows] = self._heads(self._mm(a, L["cv"]))
            c.n += rows
            c.pos = positions[part][-1]
            c.ctx.copy_(ctx[-1:])
            c.fused.copy_(fused[-1:])
            c.n_t.fill_(c.n)
            c.pos_t.fill_(c.pos)
            c.ready = True

    def _tokens(self, s: torch.Tensor) -> torch.Tensor:
        i = self._mm(s, self.head_mm).argmax(-1)
        return self.vocab[i] if self.vocab is not None else i

    def _pass(self, c: _Cache, depth: int) -> torch.Tensor:
        """The block pass on fixed shapes (whole cache buffer, device-side mask): drafted token ids [depth]."""

        b, dev = self.block, self.embed.device
        jb = torch.arange(b, device=dev)
        rest = self._mm(torch.cat([self.mask.to(self.dt)[None], c.fused], -1), self.fc)
        z = torch.cat([c.ctx, rest.expand(b - 1, -1)], 0)                               # [B, D]
        pos = c.pos_t + jb
        rel = c.rows - c.n_t                                                             # key row - first block row
        mask = (rel < 0)[None, :] | ((rel[None, :] >= 1) & (rel[None, :] <= jb[:, None]))  # context; block rows 1..j
        idx = c.n_t + jb
        first = None
        for i, L in enumerate(self.L):
            if i and self.spine:                                                         # row j reads row j-1
                sn, sa, su = self.spine[i - 1]
                z = torch.cat([z[:1], z[1:] + self._mm(F.silu(self._mm(self._rms(z[:-1], sn), sa)), su)], 0)
            a = self._rms(z, L["n1"])
            q = self._rope(self._heads(self._mm(a, L["q"])), pos)
            c.k[i].index_copy_(1, idx, self._rope(self._heads(self._mm(a, L["k"])), pos))      # in place, past the context
            c.v[i].index_copy_(1, idx, self._heads(self._mm(a, L["v"])))
            att = F.scaled_dot_product_attention(q[None], c.k[i][None], c.v[i][None], attn_mask=mask[None, None])[0]
            z = z + self._mm(att.transpose(0, 1).reshape(b, -1), L["o"])
            m = self._rms(z, L["n2"])
            z = z + self._mm(F.silu(self._mm(m, L["gate"])) * self._mm(m, L["up"]), L["down"])
            if i == 0:
                first = z[:1]
        out = torch.cat([self._rms(first, self.out_norm), self._rms(z[1:], self.out_norm)], 0)
        toks, states = [self._tokens(out[:1])], [out[:1]]
        for j in range(1, depth):
            s = out[j : j + 1]
            if self.pred is not None:                                                    # the token drafted at j-1
                ph, pe, po = self.pred
                prev = self.embed[toks[-1]].to(self.dt)
                s = s + self._mm(F.silu(self._mm(s, ph) + self._mm(prev, pe)), po)
            states.append(s)
            toks.append(self._tokens(s))
        self.states = states
        return torch.cat(toks)

    @torch.no_grad()
    def chain(self, sid: int, depth: int | None = None) -> list[int]:
        """Up to ``depth`` drafts after the stream's last kept row, from one block pass; one host sync. On CUDA with
        TENSORFOLD_KOLIBRI_BLOCK_GRAPH=1 the pass is captured once per stream and depth and replayed."""

        c = self.cache.get(sid)
        depth = min(self.block, self.depth if depth is None else depth)
        if c is None or not c.ready or depth < 1:
            return []
        if not (self.embed.is_cuda and os.environ.get("TENSORFOLD_KOLIBRI_BLOCK_GRAPH") == "1"):
            return self._pass(c, depth).tolist()
        g = c.graphs.get(depth)
        if g is None:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):                    # warm up outside the capture (allocator, autotuning)
                self._pass(c, depth)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = self._pass(c, depth)
            g = c.graphs[depth] = (graph, out)
        g[0].replay()
        return g[1].tolist()

    @torch.no_grad()
    def chain_many(self, sids: list[int], depth: int | None = None) -> dict[int, list[int]]:
        """Drafts for several streams from one pass: every projection reads its weights once for all of them; the
        attention runs per stream over its own context."""

        cs = [(sid, self.cache.get(sid)) for sid in sids]
        cs = [(sid, c) for sid, c in cs if c is not None and c.ready]
        depth = min(self.block, self.depth if depth is None else depth)
        if not cs or depth < 1:
            return {sid: [] for sid in sids}
        b, dev, S = self.block, self.embed.device, len(cs)
        jb = torch.arange(b, device=dev)
        ctx = torch.cat([c.ctx for _, c in cs])                                          # [S, D]
        fused = torch.cat([c.fused for _, c in cs])
        rest = self._mm(torch.cat([self.mask.to(self.dt)[None].expand(S, -1), fused], -1), self.fc)
        z = torch.cat([ctx[:, None], rest[:, None].expand(S, b - 1, -1)], 1).reshape(S * b, -1)
        pos = torch.cat([c.pos_t + jb for _, c in cs])
        masks = [((c.rows - c.n_t) < 0)[None, :] | (((c.rows - c.n_t)[None, :] >= 1) & ((c.rows - c.n_t)[None, :] <= jb[:, None]))
                 for _, c in cs]
        first = None
        for i, L in enumerate(self.L):
            if i and self.spine:
                sn, sa, su = self.spine[i - 1]
                zz = z.view(S, b, -1)
                inj = self._mm(F.silu(self._mm(self._rms(zz[:, :-1].reshape(S * (b - 1), -1), sn), sa)), su)
                z = torch.cat([zz[:, :1], zz[:, 1:] + inj.view(S, b - 1, -1)], 1).reshape(S * b, -1)
            a = self._rms(z, L["n1"])
            q = self._rope(self._heads(self._mm(a, L["q"])), pos)                       # [H, S*B, Dh]
            k = self._rope(self._heads(self._mm(a, L["k"])), pos)
            v = self._heads(self._mm(a, L["v"]))
            att = []
            for j, (_, c) in enumerate(cs):
                rows = slice(j * b, (j + 1) * b)
                idx = c.n_t + jb
                c.k[i].index_copy_(1, idx, k[:, rows])
                c.v[i].index_copy_(1, idx, v[:, rows])
                att.append(F.scaled_dot_product_attention(q[None, :, rows], c.k[i][None], c.v[i][None],
                                                          attn_mask=masks[j][None, None])[0])
            att = torch.cat(att, 1)                                                      # [H, S*B, Dh]
            z = z + self._mm(att.transpose(0, 1).reshape(S * b, -1), L["o"])
            m = self._rms(z, L["n2"])
            z = z + self._mm(F.silu(self._mm(m, L["gate"])) * self._mm(m, L["up"]), L["down"])
            if i == 0:
                first = z.view(S, b, -1)[:, 0]
        out = torch.cat([self._rms(first, self.out_norm)[:, None],
                         self._rms(z.view(S, b, -1)[:, 1:].reshape(S * (b - 1), -1), self.out_norm).view(S, b - 1, -1)], 1)
        toks = [self._tokens(out[:, 0])]
        for j in range(1, depth):
            s = out[:, j]
            if self.pred is not None:
                ph, pe, po = self.pred
                s = s + self._mm(F.silu(self._mm(s, ph) + self._mm(self.embed[toks[-1]].to(self.dt), pe)), po)
            toks.append(self._tokens(s))
        got = torch.stack(toks, 1).tolist()                                              # one host sync
        res = {sid: [] for sid in sids}
        res.update({sid: got[j] for j, (sid, _) in enumerate(cs)})
        return res
