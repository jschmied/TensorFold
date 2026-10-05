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


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * w).to(torch.bfloat16)


def _rope(x: torch.Tensor, pos: torch.Tensor, theta: float) -> torch.Tensor:
    """Neox halves over the last dim of x [H, T, Dh] at positions pos [T]."""

    d = x.shape[-1]
    inv = 1.0 / theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    ang = pos.float()[:, None] * inv[None]
    cos, sin = torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)
    rot = torch.cat([-x[..., d // 2:], x[..., :d // 2]], -1)
    return (x.float() * cos + rot.float() * sin).to(torch.bfloat16)


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
        g = lambda n: state[n].to(dev, torch.bfloat16).contiguous()
        self.fc = g("fc.weight")
        self.fuse = g("fuse.weight") if "fuse.weight" in state else None
        self.blocks = [{k: g(f"blocks.{i}.{k}.weight") for k in ("q", "k", "v", "o", "gate", "up", "down")}
                       | {"n1": state[f"blocks.{i}.n1.w"].to(dev, torch.float32),
                          "n2": state[f"blocks.{i}.n2.w"].to(dev, torch.float32)} for i in range(self.layers)]
        self.out_norm = state["out_norm.w"].to(dev, torch.float32)
        self.embed = embed
        ids = json.loads(Path(vocab).read_text()) if vocab else ck.get("vocab")
        self.vocab = torch.tensor(ids, device=dev) if ids is not None else None
        self.head = head[self.vocab].contiguous() if self.vocab is not None else head
        self.depth = depth
        self.cache: dict[int, dict] = {}                     # stream -> entries' (k, v) a layer, last position, output

    def drop(self, sid: int) -> None:
        self.cache.pop(sid, None)

    def _layer(self, i: int, z: torch.Tensor, pos: torch.Tensor, keys, vals, mask):
        """One block over rows z [T, D] at pos; keys/vals: the earlier keys a query sees, ahead of its own rows'."""

        b = self.blocks[i]
        t = z.shape[0]
        a = _rms(z, b["n1"], self.eps)
        q = _rope((a @ b["q"].T).view(t, self.h, self.hd).transpose(0, 1), pos, self.theta)
        k = _rope((a @ b["k"].T).view(t, self.h, self.hd).transpose(0, 1), pos, self.theta)
        v = (a @ b["v"].T).view(t, self.h, self.hd).transpose(0, 1)
        kk, vv = torch.cat([keys, k], 1), torch.cat([vals, v], 1)
        att = F.scaled_dot_product_attention(q[None], kk[None], vv[None], attn_mask=mask)[0]
        z = z + att.transpose(0, 1).reshape(t, -1) @ b["o"].T
        m = _rms(z, b["n2"], self.eps)
        return z + (F.silu(m @ b["gate"].T) * (m @ b["up"].T)) @ b["down"].T, k, v

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
            empty = torch.empty(self.h, 0, self.hd, device=dev, dtype=torch.bfloat16)
            c = self.cache[sid] = {"k": [empty] * self.layers, "v": [empty] * self.layers, "out": None, "pos": -1}
        pos = torch.tensor(positions, device=dev)
        z = self._input(torch.tensor(nxt, device=dev), torch.cat(taps, -1))
        n, old = len(positions), c["k"][0].shape[1]
        mask = torch.ones(n, old + n, dtype=torch.bool, device=dev)
        mask[:, old:] = torch.ones(n, n, dtype=torch.bool, device=dev).tril()
        for i in range(self.layers):
            z, k, v = self._layer(i, z, pos, c["k"][i], c["v"][i], mask)
            c["k"][i] = torch.cat([c["k"][i], k], 1)[:, -WINDOW:]
            c["v"][i] = torch.cat([c["v"][i], v], 1)[:, -WINDOW:]
        c["out"], c["pos"] = _rms(z[-1:], self.out_norm, self.eps), positions[-1]

    def _token(self, f: torch.Tensor) -> int:
        i = int((f @ self.head.T).argmax(-1)[0])
        return int(self.vocab[i]) if self.vocab is not None else i

    @torch.no_grad()
    def chain(self, sid: int, depth: int | None = None) -> list[int]:
        """Up to ``depth`` drafts after the stream's last entry (its output guesses the first)."""

        c = self.cache.get(sid)
        depth = self.depth if depth is None else depth
        if c is None or c["out"] is None or depth < 1:
            return []
        dev = self.embed.device
        f, out = c["out"], [self._token(c["out"])]
        self.states = [f]                                    # the chain's states (tests read them)
        ks, vs = list(c["k"]), list(c["v"])
        for j in range(1, depth):
            z = self._input(torch.tensor([out[-1]], device=dev), f)
            pos = torch.tensor([c["pos"] + j], device=dev)
            for i in range(self.layers):
                z, k, v = self._layer(i, z, pos, ks[i], vs[i], None)
                ks[i], vs[i] = torch.cat([ks[i], k], 1), torch.cat([vs[i], v], 1)
            f = _rms(z, self.out_norm, self.eps)
            self.states.append(f)
            out.append(self._token(f))
        return out
