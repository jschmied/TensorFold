"""Fixture for kolibri1/pack.zig's layer-0 test: weights.load's device arrays of layer 0, read back as raw bytes."""
import argparse
import dataclasses
from pathlib import Path

import torch


def main() -> None:
    from tensorfold.families.kolibri1.cuda import weights

    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    a = ap.parse_args()
    read = weights.Config.read
    weights.Config.read = classmethod(lambda cls, d: dataclasses.replace(read(d), layers=1))  # layer 0 only
    w = weights.load(a.model)
    l = w.layers[0]
    ex = l.experts
    arrays = {"qkv_w8": l.qkv.w8, "qkv_bs": l.qkv.bs, "o_w8": l.o.w8, "o_bs": l.o.bs, "up": ex.up, "down": ex.down,
              "up_scale": ex.up_scale, "down_scale": ex.down_scale, "input_norm": l.input_norm, "q_norm": l.q_norm,
              "k_norm": l.k_norm, "post_attn_norm": l.post_attn_norm, "pre_moe_norm": l.pre_moe_norm,
              "router": l.router, "bias": l.bias}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, t in arrays.items():
        (out / f"{name}.bin").write_bytes(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
        print(name, tuple(t.shape), t.dtype)


if __name__ == "__main__":
    main()
