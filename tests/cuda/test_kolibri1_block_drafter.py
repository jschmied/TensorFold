"""Kolibri 1's block drafter in the engine: replies with its drafts equal serial replies, in bf16, with FP8 weights
and an NVFP4 head slice, and with several streams drafted in one pass."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

sys.path.insert(0, str(Path(__file__).parent))
from kolibri1_tiny import write  # noqa: E402

TAPS = [0, 2]
CFG = {"hidden": 256, "heads": 2, "head_dim": 128, "ffn": 384, "layers": 2, "taps": len(TAPS), "block": 4,
       "eps": 1e-6, "theta": 10000.0, "chain_ctx": True, "row0_chain": True, "pred_rank": 64, "spine_rank": 64}


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    from tensorfold.families.kolibri1.cuda.weights import load

    return load(write(tmp_path_factory.mktemp("kolibri1-tiny")))


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> Path:
    """A random block drafter in train_block.py's checkpoint format (model_block names)."""

    g = torch.Generator().manual_seed(1)
    d, hd, f, r = CFG["hidden"], CFG["heads"] * CFG["head_dim"], CFG["ffn"], CFG["pred_rank"]

    def lin(n, k):
        return torch.randn(n, k, generator=g) / math.sqrt(k)

    def norm():
        return 1 + 0.1 * torch.randn(d, generator=g)

    state = {"fuse.weight": lin(d, len(TAPS) * d), "fc.weight": lin(d, 2 * d), "mask": 0.02 * torch.randn(d, generator=g),
             "out_norm.w": norm(), "pred_h.weight": lin(r, d), "pred_e.weight": lin(r, d), "pred_out.weight": lin(d, r)}
    for i in range(CFG["layers"]):
        p = f"layers.{i}."
        for k, (n, kk) in {"q": (hd, d), "k": (hd, d), "v": (hd, d), "ck": (hd, d), "cv": (hd, d), "o": (d, hd),
                           "gate": (f, d), "up": (f, d), "down": (d, f)}.items():
            state[p + k + ".weight"] = lin(n, kk)
        for k in ("n1", "n2", "nc"):
            state[p + k + ".w"] = norm()
    for i in range(CFG["layers"] - 1):
        state[f"spine_n.{i}.w"] = norm()
        state[f"spine_a.{i}.weight"] = lin(CFG["spine_rank"], d)
        state[f"spine_u.{i}.weight"] = lin(d, CFG["spine_rank"])
    path = tmp_path_factory.mktemp("block") / "drafter.pt"
    torch.save({"config": CFG, "state": state, "taps": TAPS}, path)
    (path.parent / "vocab.json").write_text(json.dumps(list(range(0, 512, 2))))
    return path


def tokens(n: int, seed: int = 0) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 500, (n,), generator=g).tolist()


def decode(weights, checkpoint, prompts, sampling, *, drafter, quant, monkeypatch):
    from tensorfold.cuda.streams import Stream
    from tensorfold.families.kolibri1.cuda import decoder
    from tensorfold.families.kolibri1.cuda.block_drafter import BlockDrafter
    from tensorfold.families.kolibri1.cuda.forward import Model

    monkeypatch.setattr(decoder.CopyIndex, "propose", lambda self, context, most: [])   # copies never fire
    dr = None
    if drafter:
        dr = BlockDrafter(checkpoint, weights.embed, weights.head, depth=3, vocab=checkpoint.parent / "vocab.json",
                          quant=quant[0], head_quant=quant[1])
    d = decoder.Decoder(Model(weights, 512, len(prompts)), (511,), drafter=dr)
    streams = [Stream(list(p), 36, sampling, draft=drafter) for p in prompts]
    for s in streams:
        d.admit(s)
    while not all(s.done for s in streams):
        d.finish(d.round())
    return [list(s.out) for s in streams], d.learned_rows


@pytest.mark.parametrize("quant", [("bf16", "bf16"), ("fp8", "fp8"), ("fp8", "nvfp4")])
def test_block_drafts_equal_the_serial_reply(weights, checkpoint, monkeypatch, quant):
    from tensorfold.engine.exact_sampling import Sampling

    prompts = [tokens(60, seed=3), tokens(45, seed=4), tokens(52, seed=8)]     # three streams: one batched pass
    for sampling in (None, Sampling(seed=5, temperature=0.8, top_k=20, top_p=0.95)):
        together, rows = decode(weights, checkpoint, prompts, sampling, drafter=True, quant=quant,
                                monkeypatch=monkeypatch)
        assert rows > 0                                                         # learned drafts were verified
        for p, got in zip(prompts, together):
            serial, _ = decode(weights, checkpoint, [p], sampling, drafter=False, quant=quant,
                               monkeypatch=monkeypatch)
            assert got == serial[0]
