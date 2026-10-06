"""Kolibri 1's learned drafter in the engine: its chain is the trained rollout's, and drafted replies equal serial."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
sys.path.insert(0, str(Path(__file__).parent))

from kolibri1_tiny import write

TAPS = [0, 2]
CFG = {
    "hidden": 256,
    "heads": 2,
    "head_dim": 64,
    "ffn": 384,
    "eps": 1e-6,
    "theta": 10000.0,
    "layers": 1,
    "taps": len(TAPS),
}


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    from tensorfold.families.kolibri1.cuda.weights import load

    return load(write(tmp_path_factory.mktemp("kolibri1-tiny")))


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> Path:
    """A random drafter in the training script's checkpoint format (model_mt names)."""

    g = torch.Generator().manual_seed(0)
    d, hd, f = CFG["hidden"], CFG["heads"] * CFG["head_dim"], CFG["ffn"]

    def lin(n, k):
        return torch.randn(n, k, generator=g) / math.sqrt(k)

    state = {
        "fc.weight": lin(d, 2 * d),
        "fuse.weight": lin(d, len(TAPS) * d),
        "out_norm.w": 1 + 0.1 * torch.randn(d, generator=g),
        "blocks.0.n1.w": 1 + 0.1 * torch.randn(d, generator=g),
        "blocks.0.n2.w": 1 + 0.1 * torch.randn(d, generator=g),
        "blocks.0.q.weight": lin(hd, d),
        "blocks.0.k.weight": lin(hd, d),
        "blocks.0.v.weight": lin(hd, d),
        "blocks.0.o.weight": lin(d, hd),
        "blocks.0.gate.weight": lin(f, d),
        "blocks.0.up.weight": lin(f, d),
        "blocks.0.down.weight": lin(d, f),
    }
    path = tmp_path_factory.mktemp("drafter") / "drafter.pt"
    torch.save({"config": CFG, "state": state, "taps": TAPS}, path)
    return path


def tokens(n: int, seed: int = 0) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 500, (n,), generator=g).tolist()


def reference(state, embed, head, feats, nxt, drafts):
    """The training rollout (model_mt.Drafter.rollout) for the last row's chain, fp32: its states, step by step."""

    s = {k: v.float().cuda() for k, v in state.items()}
    h, hd, eps, theta = CFG["heads"], CFG["head_dim"], CFG["eps"], CFG["theta"]

    def rms(x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

    def rope(x, pos):
        dd = x.shape[-1]
        inv = 1.0 / theta ** (torch.arange(0, dd, 2, device=x.device, dtype=torch.float32) / dd)
        ang = pos.float()[:, None] * inv[None]
        cos, sin = torch.cat([ang.cos()] * 2, -1), torch.cat([ang.sin()] * 2, -1)
        return x * cos + torch.cat([-x[..., dd // 2 :], x[..., : dd // 2]], -1) * sin

    def block(z, pos, keys, vals, mask):
        a = rms(z, s["blocks.0.n1.w"])
        t = z.shape[0]
        q = rope((a @ s["blocks.0.q.weight"].T).view(t, h, hd).transpose(0, 1), pos)
        k = rope((a @ s["blocks.0.k.weight"].T).view(t, h, hd).transpose(0, 1), pos)
        v = (a @ s["blocks.0.v.weight"].T).view(t, h, hd).transpose(0, 1)
        kk, vv = torch.cat([keys, k], 1), torch.cat([vals, v], 1)
        att = F.scaled_dot_product_attention(q[None], kk[None], vv[None], attn_mask=mask)[0]
        z = z + att.transpose(0, 1).reshape(t, -1) @ s["blocks.0.o.weight"].T
        m = rms(z, s["blocks.0.n2.w"])
        return (
            z
            + (F.silu(m @ s["blocks.0.gate.weight"].T) * (m @ s["blocks.0.up.weight"].T)) @ s["blocks.0.down.weight"].T,
            k,
            v,
        )

    t = feats.shape[0]
    emb = embed.float()
    z = torch.cat([emb[nxt], feats.float() @ s["fuse.weight"].T], -1) @ s["fc.weight"].T
    empty = torch.empty(h, 0, hd, device="cuda")
    z, ks, vs = block(
        z, torch.arange(t, device="cuda"), empty, empty, torch.ones(t, t, dtype=torch.bool, device="cuda").tril()
    )
    f = rms(z[-1:], s["out_norm.w"])
    out = [f]
    for j, tok in enumerate(drafts[:-1], 1):
        z = torch.cat([emb[torch.tensor([tok], device="cuda")], f], -1) @ s["fc.weight"].T
        z, k, v = block(z, torch.tensor([t - 1 + j], device="cuda"), ks, vs, None)
        ks, vs = torch.cat([ks, k], 1), torch.cat([vs, v], 1)
        f = rms(z, s["out_norm.w"])
        out.append(f)
    return out


def test_the_chain_is_the_trained_rollout(weights, checkpoint):
    from tensorfold.families.kolibri1.cuda.drafter import Drafter
    from tensorfold.families.kolibri1.cuda.forward import Chain, Model

    ids = tokens(80, seed=2)
    _, taps = Model(weights, 256).forward([Chain(0, 0, ids)], prompt=True, features=True, taps=TAPS)
    feats = torch.cat(taps, -1)[:-1]  # rows 0..78; each row's next token is ids[r + 1]
    dr = Drafter(checkpoint, weights.embed, weights.head, depth=3)
    dr.add(7, list(range(79)), ids[1:80], [t[:-1] for t in taps])
    drafts = dr.chain(7)
    want = reference(
        torch.load(checkpoint)["state"],
        weights.embed,
        weights.head,
        feats,
        torch.tensor(ids[1:80], device="cuda"),
        drafts,
    )
    assert len(drafts) == 3
    for got, ref in zip(dr.states, want):
        assert F.cosine_similarity(got.float(), ref, dim=-1).min() > 0.999
    assert drafts[0] == int((want[0] @ weights.head.float().T).argmax())


def decode(weights, checkpoint, prompts, sampling, *, draft, drafter, monkeypatch):
    from tensorfold.cuda.streams import Stream
    from tensorfold.families.kolibri1.cuda import decoder
    from tensorfold.families.kolibri1.cuda.drafter import Drafter
    from tensorfold.families.kolibri1.cuda.forward import Model

    monkeypatch.setattr(decoder.CopyIndex, "propose", lambda self, context, most: [])  # copies never fire
    dr = Drafter(checkpoint, weights.embed, weights.head, depth=3) if drafter else None
    d = decoder.Decoder(Model(weights, 512, len(prompts)), (511,), drafter=dr)
    streams = [Stream(list(p), 36, sampling, draft=draft) for p in prompts]
    for s in streams:
        d.admit(s)
    while not all(s.done for s in streams):
        d.finish(d.round())
    return [list(s.out) for s in streams], d.learned_rows


def test_learned_drafts_equal_the_serial_reply(weights, checkpoint, monkeypatch):
    from tensorfold.engine.exact_sampling import Sampling

    prompts = [tokens(60, seed=3), tokens(45, seed=4)]
    for sampling in (None, Sampling(seed=5, temperature=0.8, top_k=20, top_p=0.95)):
        together, rows = decode(
            weights, checkpoint, prompts, sampling, draft=True, drafter=True, monkeypatch=monkeypatch
        )
        assert rows > 0  # the learned chain was verified
        for p, got in zip(prompts, together):
            serial, _ = decode(weights, checkpoint, [p], sampling, draft=False, drafter=False, monkeypatch=monkeypatch)
            assert got == serial[0]


def test_a_release_directory_loads_the_same_drafter(weights, checkpoint, tmp_path):
    import json

    from safetensors.torch import save_file

    from tensorfold.families.kolibri1.cuda.drafter import Drafter
    from tensorfold.families.kolibri1.cuda.forward import Chain, Model

    ck = torch.load(checkpoint)
    save_file({k: v.contiguous() for k, v in ck["state"].items()}, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps({"drafter": ck["config"], "taps": ck["taps"]}))
    (tmp_path / "draft_vocab.json").write_text(json.dumps(list(range(0, 512, 2))))
    ids = tokens(40, seed=6)
    _, taps = Model(weights, 128).forward([Chain(0, 0, ids)], prompt=True, features=True, taps=TAPS)
    got = []
    for src, vocab in ((tmp_path, None), (checkpoint, tmp_path / "draft_vocab.json")):
        dr = Drafter(src, weights.embed, weights.head, depth=3, vocab=vocab)
        dr.add(0, list(range(39)), ids[1:40], [t[:-1] for t in taps])
        got.append(dr.chain(0))
    assert got[0] == got[1] and all(t % 2 == 0 for t in got[0])  # same chain, drawn from the slice
