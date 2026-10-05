"""A learned drafter for Kolibri 1 (EAGLE-3 style), proposing a chain when copies from the context find none.

Each kept row r enters the drafter once, from Kolibri's tapped layer states at r fused to one and the embedding of
the token after r; it attends to the entries before it (the last ``WINDOW``, as trained). Its output, read by
Kolibri's own head, guesses the token two after r; later chain steps read the drafter's own output instead of the
fused taps and attend to the entries and to their own chain. Drafts change no reply: the target verifies them.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

WINDOW = 2048            # entries an entry attends to (the training window)
CHAIN = 8                # cache rows kept free past the entries for a chain's own steps


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * w).to(torch.bfloat16)


class _Cache:
    """One stream's entries: keys and values a layer in a buffer of 2 x WINDOW rows, appended in place."""

    def __init__(self, layers: int, h: int, hd: int, dev) -> None:
        shape = (h, 2 * WINDOW + CHAIN, hd)
        self.k = [torch.empty(shape, device=dev, dtype=torch.bfloat16) for _ in range(layers)]
        self.v = [torch.empty(shape, device=dev, dtype=torch.bfloat16) for _ in range(layers)]
        self.n, self.pos, self.out = 0, -1, None

    def room(self, rows: int) -> None:
        """Keep the last WINDOW entries at the front when ``rows`` more would not fit (one copy per WINDOW rows)."""

        if self.n + rows + CHAIN > self.k[0].shape[1]:
            keep = min(self.n, WINDOW)
            for t in self.k + self.v:
                t[:, :keep] = t[:, self.n - keep : self.n].clone()
            self.n = keep


class Drafter:
    def __init__(self, path: str | Path, embed: torch.Tensor, head: torch.Tensor, *, depth: int = 2,
                 vocab: str | Path | None = None) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        cfg = {"layers": 1, "taps": 1, **ck["config"]}
        self.d, self.h, self.hd = cfg["hidden"], cfg["heads"], cfg["head_dim"]
        self.eps, self.theta, self.layers = cfg["eps"], cfg["theta"], cfg["layers"]
        self.taps = tuple(int(t) for t in ck["taps"])
        if len(self.taps) != cfg["taps"] or embed.shape[1] != self.d:
            raise ValueError(f"drafter {path}: {cfg['taps']} taps of width {self.d} do not fit {self.taps} / "
                             f"Kolibri's width {embed.shape[1]}")
        dev = embed.device
        state = {}
        for name, t in ck["state"].items():                 # one-layer checkpoints may carry flat names
            flat = name.split(".")[0] in ("n1", "n2", "q", "k", "v", "o", "gate", "up", "down")
            state[("blocks.0." + name) if flat else name] = t

        def g(n: str) -> torch.Tensor:
            return state[n].to(dev, torch.bfloat16).contiguous()

        self.fc = g("fc.weight")
        self.fuse = g("fuse.weight") if "fuse.weight" in state else None
        self.blocks = []
        for i in range(self.layers):
            b = {k: g(f"blocks.{i}.{k}.weight") for k in ("o", "down")}
            b["qkv"] = torch.cat([g(f"blocks.{i}.{k}.weight") for k in ("q", "k", "v")]).contiguous()
            b["gate_up"] = torch.cat([g(f"blocks.{i}.{k}.weight") for k in ("gate", "up")]).contiguous()
            b["n1"], b["n2"] = (state[f"blocks.{i}.{k}.w"].to(dev, torch.float32) for k in ("n1", "n2"))
            self.blocks.append(b)
        self.ffn = self.blocks[0]["down"].shape[1]
        self.out_norm = state["out_norm.w"].to(dev, torch.float32)
        self.embed = embed
        ids = json.loads(Path(vocab).read_text()) if vocab else ck.get("vocab")
        self.vocab = torch.tensor(ids, device=dev) if ids is not None else None
        self.head = head[self.vocab].contiguous() if self.vocab is not None else head
        self.inv = 1.0 / self.theta ** (torch.arange(0, self.hd, 2, device=dev, dtype=torch.float32) / self.hd)
        self.depth = depth
        self.cache: dict[int, _Cache] = {}
        self.states: list[torch.Tensor] = []                 # the last chain's states (tests read them)

    def drop(self, sid: int) -> None:
        self.cache.pop(sid, None)

    def _rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Neox halves over the last dim of x [H, T, Dh] at positions pos [T]."""

        ang = pos.float()[:, None] * self.inv[None]
        cos, sin = ang.cos().repeat(1, 2), ang.sin().repeat(1, 2)
        half = self.hd // 2
        rot = torch.cat([-x[..., half:], x[..., :half]], -1)
        return (x.float() * cos + rot.float() * sin).to(torch.bfloat16)

    def _layer(self, i: int, z: torch.Tensor, pos: torch.Tensor, c: _Cache, at: int, mask) -> torch.Tensor:
        """Block i over rows z [T, D] at pos; their keys go to cache rows at.., queries see rows < at + T."""

        b, t = self.blocks[i], z.shape[0]
        qkv = (_rms(z, b["n1"], self.eps) @ b["qkv"].T).view(t, 3, self.h, self.hd).transpose(0, 2)  # [H, 3, T, Dh]
        q, k = self._rope(qkv[:, 0], pos), self._rope(qkv[:, 1], pos)
        c.k[i][:, at : at + t], c.v[i][:, at : at + t] = k, qkv[:, 2]
        keys, vals = c.k[i][None, :, : at + t], c.v[i][None, :, : at + t]
        att = F.scaled_dot_product_attention(q[None], keys, vals, attn_mask=mask)[0]
        z = z + att.transpose(0, 1).reshape(t, -1) @ b["o"].T
        gu = _rms(z, b["n2"], self.eps) @ b["gate_up"].T
        return z + (F.silu(gu[:, : self.ffn]) * gu[:, self.ffn :]) @ b["down"].T

    def _input(self, nxt: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        if self.fuse is not None and feats.shape[-1] != self.d:
            feats = feats.to(torch.bfloat16) @ self.fuse.T
        return torch.cat([self.embed[nxt], feats.to(torch.bfloat16)], -1) @ self.fc.T

    @torch.no_grad()
    def add(self, sid: int, positions: list[int], nxt: list[int], taps: list[torch.Tensor]) -> None:
        """Entries for kept rows at ``positions`` (in order): the token after each and the rows' tapped states."""

        if not positions:
            return
        dev = self.embed.device
        c = self.cache.get(sid)
        if c is None:
            c = self.cache[sid] = _Cache(self.layers, self.h, self.hd, dev)
        for lo in range(0, len(positions), WINDOW):           # a long prompt tail in window-sized parts
            part = slice(lo, lo + WINDOW)
            n = len(positions[part])
            c.room(n)
            pos = torch.tensor(positions[part], device=dev)
            z = self._input(torch.tensor(nxt[part], device=dev), torch.cat([t[part] for t in taps], -1))
            mask = None
            if n > 1:                                          # every earlier entry, then causal among the new ones
                mask = torch.ones(n, c.n + n, dtype=torch.bool, device=dev)
                mask[:, c.n :] = torch.ones(n, n, dtype=torch.bool, device=dev).tril()
            for i in range(self.layers):
                z = self._layer(i, z, pos, c, c.n, mask)
            c.n += n
            c.out, c.pos = _rms(z[-1:], self.out_norm, self.eps), positions[part][-1]

    def _tokens(self, f: torch.Tensor) -> torch.Tensor:
        i = (f @ self.head.T).argmax(-1)
        return self.vocab[i] if self.vocab is not None else i

    @torch.no_grad()
    def chain(self, sid: int, depth: int | None = None) -> list[int]:
        """Up to ``depth`` drafts after the stream's last entry (its output guesses the first); one host sync."""

        c = self.cache.get(sid)
        depth = min(CHAIN, self.depth if depth is None else depth)
        if c is None or c.out is None or depth < 1:
            return []
        dev = self.embed.device
        f = c.out
        out = [self._tokens(f)]
        self.states = [f]
        for j in range(1, depth):                              # its own steps sit past the entries, uncommitted
            z = self._input(out[-1], f)
            pos = torch.full((1,), c.pos + j, device=dev, dtype=torch.long)
            for i in range(self.layers):
                z = self._layer(i, z, pos, c, c.n + j - 1, None)
            f = _rms(z, self.out_norm, self.eps)
            self.states.append(f)
            out.append(self._tokens(f))
        return torch.cat(out).tolist()
