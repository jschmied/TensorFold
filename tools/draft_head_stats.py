"""Capture a draft head's input statistics (#1870) on the pinned calibration prompts of draft_head_prompts.json.

Start the server with TF_FLASHNEXT_DRAFT_STATS_CAPTURE=<file> and --parallel 1, run this against it, then stop the
server (its stop writes the last rows). The file is keyed by the head's checksum: serve with
TF_FLASHNEXT_DRAFT_HEAD=affine4 TF_FLASHNEXT_DRAFT_FIT=weighted TF_FLASHNEXT_DRAFT_STATS=<file>.
"""
import argparse
import json
import os
import struct
import urllib.request

PROMPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "draft_head_prompts.json")


def run(base, model, tokens):
    prompts = json.load(open(PROMPTS, encoding="utf-8"))["calibration"]
    for i, p in enumerate(prompts):
        body = {"model": model, "messages": [{"role": "user", "content": p}], "max_tokens": tokens,
                "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            usage = json.loads(r.read()).get("usage") or {}
        print(f"calibration {i + 1}/{len(prompts)}: {usage.get('completion_tokens')} tokens", flush=True)


def describe(path):
    """The header of a statistics file: width, head checksum, rows."""
    with open(path, "rb") as f:
        head = f.read(32)
    if head[:8] != b"TFDHSTAT":
        raise SystemExit(f"{path}: not a draft head statistics file")
    version, width, checksum, rows = struct.unpack("<IIQQ", head[8:])
    return {"version": version, "width": width, "checksum": f"{checksum:016x}", "rows": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="m")
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--describe", help="print a statistics file's header and exit")
    a = ap.parse_args()
    if a.describe:
        print(json.dumps(describe(a.describe)))
        return
    run(a.base, a.model, a.tokens)


if __name__ == "__main__":
    main()
