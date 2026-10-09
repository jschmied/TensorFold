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
                 vocab: str | Path | None = None, dtype: torch.dtype = torch.bfloat16) -> None:
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
            fused = torch.cat([t[part] for t in taps], -1).to(self.dt) @ self.fuse.T
            ctx = torch.cat([self.embed[torch.tensor(nxt[part], device=dev)].to(self.dt), fused], -1) @ self.fc.T
            for i, L in enumerate(self.L):
                a = self._rms(ctx, L["nc"])
                c.k[i][:, c.n : c.n + rows] = self._rope(self._heads(a @ L["ck"].T), pos)
                c.v[i][:, c.n : c.n + rows] = self._heads(a @ L["cv"].T)
            c.n += rows
            c.pos = positions[part][-1]
            c.ctx.copy_(ctx[-1:])
            c.fused.copy_(fused[-1:])
            c.n_t.fill_(c.n)
            c.pos_t.fill_(c.pos)
            c.ready = True

    def _tokens(self, s: torch.Tensor) -> torch.Tensor:
        i = (s @ self.head.T).argmax(-1)
        return self.vocab[i] if self.vocab is not None else i

    def _pass(self, c: _Cache, depth: int) -> torch.Tensor:
        """The block pass on fixed shapes (whole cache buffer, device-side mask): drafted token ids [depth]."""

        b, dev = self.block, self.embed.device
        jb = torch.arange(b, device=dev)
        rest = torch.cat([self.mask.to(self.dt)[None], c.fused], -1) @ self.fc.T
        z = torch.cat([c.ctx, rest.expand(b - 1, -1)], 0)                               # [B, D]
        pos = c.pos_t + jb
        rel = c.rows - c.n_t                                                             # key row - first block row
        mask = (rel < 0)[None, :] | ((rel[None, :] >= 1) & (rel[None, :] <= jb[:, None]))  # context; block rows 1..j
        idx = c.n_t + jb
        first = None
        for i, L in enumerate(self.L):
            if i and self.spine:                                                         # row j reads row j-1
                sn, sa, su = self.spine[i - 1]
                z = torch.cat([z[:1], z[1:] + F.silu(self._rms(z[:-1], sn) @ sa.T) @ su.T], 0)
            a = self._rms(z, L["n1"])
            q = self._rope(self._heads(a @ L["q"].T), pos)
            c.k[i].index_copy_(1, idx, self._rope(self._heads(a @ L["k"].T), pos))      # in place, past the context
            c.v[i].index_copy_(1, idx, self._heads(a @ L["v"].T))
            att = F.scaled_dot_product_attention(q[None], c.k[i][None], c.v[i][None], attn_mask=mask[None, None])[0]
            z = z + att.transpose(0, 1).reshape(b, -1) @ L["o"].T
            m = self._rms(z, L["n2"])
            z = z + (F.silu(m @ L["gate"].T) * (m @ L["up"].T)) @ L["down"].T
            if i == 0:
                first = z[:1]
        out = torch.cat([self._rms(first, self.out_norm), self._rms(z[1:], self.out_norm)], 0)
        toks, states = [self._tokens(out[:1])], [out[:1]]
        for j in range(1, depth):
            s = out[j : j + 1]
            if self.pred is not None:                                                    # the token drafted at j-1
                ph, pe, po = self.pred
                prev = self.embed[toks[-1]].to(self.dt)
                s = s + F.silu(s @ ph.T + prev @ pe.T) @ po.T
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
