#!/usr/bin/env python3
"""LFM3 float64 CPU reference in numpy only (MLX 4-bit dequantized exactly), with per-layer dumps for the XPU forward."""
# usage:
#   ref64_np.py dump MODEL_DIR OUT_DIR      the 3 fixed prompts: ids, hidden state after every layer (f32), logits rows
#   ref64_np.py check MODEL_DIR TOKENS_JSON teacher-forced float64 top-1 of {"name": {"prompt": [...], "reply": [...]}}
# The math follows modeling_lfm3.py (as tools/lfm3/ref64.py of #1508): sandwich norms, gated conv7, QK norm, zero-tail
# partial RoPE, a 256-key window on sliding layers, a sigmoid gate per head, embedding_norm before the tied head.
# The bf16 pass rounds where the XPU kernels store bf16 (attention P stays unrounded, logits fp32): the yardstick.
import json
import os
import struct
import sys

import numpy as np

F64 = np.float64


def bf16(x):
    """Nearest bf16 (ties to even) of float64 values, kept as float64."""
    m, e = np.frexp(x)
    return np.ldexp(np.round(m * 256.0), e - 8)


def same(x):
    return x


class Model:
    def __init__(self, d):
        self.cfg = json.load(open(f"{d}/config.json"))
        path = f"{d}/model.safetensors"
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.hdr = json.loads(f.read(n))
        self.hdr.pop("__metadata__", None)
        self.base = 8 + n
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        self.prefix = "language_model.model."
        self.cache = {}

    def raw(self, k):
        h = self.hdr[self.prefix + k]
        a, b = h["data_offsets"]
        buf = self.mm[self.base + a: self.base + b]
        if h["dtype"] == "BF16":
            return (buf.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(h["shape"]).astype(F64)
        if h["dtype"] == "U32":
            return buf.view(np.uint32).reshape(h["shape"])
        raise ValueError(h["dtype"])

    def vec(self, k):
        return self.raw(k)

    def lin(self, name, rows=None):
        """MLX affine 4-bit [out, in] in float64: q * scale + bias per group of 64, nibble j of a word is element j."""
        w, s, b = self.raw(name + ".weight"), self.raw(name + ".scales"), self.raw(name + ".biases")
        if rows is not None:
            w, s, b = w[rows], s[rows], b[rows]
        q = (w[..., None] >> (np.arange(8, dtype=np.uint32) * 4)) & 15
        q = q.reshape(w.shape[0], s.shape[1], 64).astype(F64)
        return (q * s[..., None] + b[..., None]).reshape(w.shape[0], -1)

    def forward(self, ids, bf, dump=None):
        """Logits [T, V] in float64; dump(L, h) gets the hidden state after each layer, dump(-1, x) the final norm."""
        cfg, T = self.cfg, len(ids)
        D, H, HK, hd = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        eps = cfg["norm_eps"]

        def rms(h, w):
            return bf(w * bf(h / np.sqrt((h * h).mean(-1, keepdims=True) + eps)))

        def mm(x, name):
            return x @ self.lin(name).T

        tabs = {}
        for kind, rp in cfg["rope_parameters"].items():
            angles = int(rp.get("partial_rotary_factor", 1.0) * hd) // 2
            inv = 1.0 / rp["rope_theta"] ** (np.arange(0, 2 * angles, 2, dtype=F64) / hd)
            inv = np.concatenate([inv, np.zeros(hd // 2 - angles)])
            ang = np.arange(T, dtype=F64)[:, None] * inv[None]
            tabs[kind] = bf(np.concatenate([np.cos(ang)] * 2, -1)), bf(np.concatenate([np.sin(ang)] * 2, -1))
        i, j = np.arange(T)[:, None], np.arange(T)[None]
        masks = {"full_attention": j > i, "sliding_attention": (j > i) | (i - j >= cfg["sliding_window"])}
        h = bf(self.lin("embed_tokens", rows=np.array(ids)))
        for L, kind in enumerate(cfg["layer_types"]):
            p = f"layers.{L}."
            x = rms(h, self.vec(p + "operator_norm.weight"))
            if kind == "conv":
                B, C, xx = np.split(bf(mm(x, p + "conv.in_proj")), 3, -1)
                bx = bf(B * xx)
                w = self.vec(p + "conv.conv.weight").reshape(D, 7)
                pad = np.concatenate([np.zeros((6, D)), bx])
                acc = np.zeros((T, D))
                for k in range(7):
                    acc = acc + pad[k:k + T] * w[:, k]
                o = bf(mm(bf(C * bf(acc)), p + "conv.out_proj"))
            else:
                q, k, v = (bf(mm(x, p + f"self_attn.{n}_proj")) for n in "qkv")
                g = bf(mm(x, p + "self_attn.gate_proj"))
                q = rms(q.reshape(T, H, hd), self.vec(p + "self_attn.q_norm.weight")).transpose(1, 0, 2)
                k = rms(k.reshape(T, HK, hd), self.vec(p + "self_attn.k_norm.weight")).transpose(1, 0, 2)
                cos, sin = tabs[kind]

                def rot(t):
                    return t * cos + np.concatenate([-t[..., hd // 2:], t[..., : hd // 2]], -1) * sin

                q, k = bf(rot(q)), bf(rot(k))
                k = np.repeat(k, H // HK, 0)
                v = np.repeat(v.reshape(T, HK, hd).transpose(1, 0, 2), H // HK, 0)
                s = (q @ k.transpose(0, 2, 1)) / np.sqrt(hd)
                s[:, masks[kind]] = -np.inf
                e = np.exp(s - s.max(-1, keepdims=True))
                a = bf((e @ v) / e.sum(-1, keepdims=True)).transpose(1, 0, 2)
                a = bf(a * bf(1.0 / (1.0 + np.exp(-g)))[..., None])
                o = bf(mm(a.reshape(T, H * hd), p + "self_attn.o_proj"))
            h = bf(h + rms(o, self.vec(p + "post_operator_norm.weight")))
            x = rms(h, self.vec(p + "ffn_norm.weight"))
            f = p + "feed_forward."
            a1 = bf(mm(x, f + "w1"))
            act = bf(bf(a1 / (1.0 + np.exp(-a1))) * bf(mm(x, f + "w3")))
            h = bf(h + rms(bf(mm(act, f + "w2")), self.vec(p + "post_ffn_norm.weight")))
            if dump:
                dump(L, h)
        x = rms(h, self.vec("embedding_norm.weight"))
        if dump:
            dump(-1, x)
        V = cfg["vocab_size"]
        return np.concatenate([x @ self.lin("embed_tokens", rows=np.arange(a, min(a + 8000, V))).T
                               for a in range(0, V, 8000)], -1)


def prompts():
    """Three of #1508's ref64.py prompts (the items prompt with 60 items: longer than the 256-key window)."""
    colors = "red green blue amber violet".split()
    items = " ".join(f"Item {i}: the {colors[i % 5]} box holds {7 * i % 23} stones." for i in range(60))
    return [("capitals", "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is"),
            ("chat", "<|im_start|>user\nExplain why the sky is blue in two sentences.<|im_end|>\n<|im_start|>assistant\n"),
            ("items", items + " Question: how many stones does item 3 hold? Answer: item 3 holds")]


def rel_rows(a, b):
    """Per-row ||a - b|| / ||b||: mean and max."""
    r = np.linalg.norm(a - b, axis=-1) / np.linalg.norm(b, axis=-1)
    return float(r.mean()), float(r.max())


def dump_all(model_dir, out):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
    m = Model(model_dir)
    bos = m.cfg["bos_token_id"]
    summary = {}
    for name, text in prompts():
        ids = [bos] + tok.encode(text, add_special_tokens=False).ids
        d = f"{out}/{name}"
        os.makedirs(d, exist_ok=True)
        hs = {}

        def keep(L, h):
            hs[L] = h.copy()
            h.astype(np.float32).tofile(f"{d}/h{L:02d}.f32" if L >= 0 else f"{d}/final.f32")

        lf = m.forward(ids, same, keep)
        f64h = hs
        hs = {}
        lb = m.forward(ids, bf16, lambda L, h: hs.__setitem__(L, h.copy()))
        gap = {L: rel_rows(hs[L], f64h[L]) for L in f64h}
        top = lf.argmax(-1)
        np.asarray(top, dtype=np.int32).tofile(f"{d}/top1.i32")
        lf[-1].astype(np.float32).tofile(f"{d}/logits_last.f32")
        json.dump({"ids": ids, "T": len(ids), "top1": top.tolist(),
                   "bf16_gap": {str(L): v for L, v in gap.items()},
                   "bf16_top1_agree": int((lb.argmax(-1) == top).sum())}, open(f"{d}/meta.json", "w"))
        summary[name] = {"T": len(ids), "bf16_gap_max": max(v[1] for v in gap.values()),
                         "next": int(top[-1]), "bf16_top1_agree": f"{int((lb.argmax(-1) == top).sum())}/{len(ids)}"}
        print(name, json.dumps(summary[name]), flush=True)
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)


def check(model_dir, path):
    """Teacher-forced: the float64 top-1 after each prefix of prompt + reply against the reply's own next token."""
    m = Model(model_dir)
    res = {}
    for name, c in json.load(open(path)).items():
        P, reply = len(c["prompt"]), c["reply"]
        lf = m.forward(c["prompt"] + reply[:-1], same)
        picks = lf[P - 1:].argmax(-1)
        srt = np.sort(lf[P - 1:], -1)
        miss = [{"j": j, "xpu": int(reply[j]), "f64": int(picks[j]), "margin": float(srt[j, -1] - lf[P - 1 + j, reply[j]])}
                for j in range(len(reply)) if picks[j] != reply[j]]
        res[name] = {"agree": f"{len(reply) - len(miss)}/{len(reply)}", "misses": miss}
        print(name, json.dumps(res[name]), file=sys.stderr, flush=True)
    return res


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "dump":
        dump_all(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 4 and sys.argv[1] == "check":
        json.dump(check(sys.argv[2], sys.argv[3]), sys.stdout, indent=1)
    else:
        sys.exit(__doc__)
