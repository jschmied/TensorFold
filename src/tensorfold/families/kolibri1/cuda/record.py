"""Records what serving computes for drafter training: each kept row's token, tapped layer states and top-k choices."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

HEAD_ROWS = 1024         # rows a top-k pass takes (the head's 128k-wide logits)


class Recorder:
    """Per request, append-only files under ``root``: tokens, positions, states (bf16), top-k ids and log-probs."""

    def __init__(self, root: str | Path, taps: Sequence[int], k: int = 32, floor_gb: float = 25.0) -> None:
        self.root = Path(root) / time.strftime("%Y%m%d")
        self.root.mkdir(parents=True, exist_ok=True)
        self.taps, self.k, self.floor = tuple(int(t) for t in taps), int(k), floor_gb * 2**30
        self.rows: dict[int, int] = {}
        self.paused = False
        self.dims = 0

    def _room(self) -> bool:
        free = shutil.disk_usage(self.root).free
        self.paused = free < self.floor
        return not self.paused

    def _files(self, sid: int) -> Path:
        return self.root / f"{os.getpid()}-{sid}"

    def add(self, sid: int, positions: Sequence[int], tokens: Sequence[int], states: Sequence[torch.Tensor],
            final: torch.Tensor, head: torch.Tensor) -> None:
        """Rows of one request in position order: their input tokens, tapped states [n, D] each, the head's top-k."""

        from tensorfold.cuda import moe as shared

        if not len(tokens) or not self._room():
            return
        ids, logp = [], []
        for i in range(0, final.shape[0], HEAD_ROWS):
            logits = shared.router(final[i:i + HEAD_ROWS].contiguous(), head)
            top = torch.topk(logits, self.k, dim=-1)
            ids.append(top.indices.to(torch.int32))
            logp.append((top.values - torch.logsumexp(logits, -1, keepdim=True)).to(torch.float16))
        base = self._files(sid)
        cols = torch.cat(list(states), -1).contiguous()
        self.dims = int(final.shape[1])
        for ext, arr in ((".tok", np.asarray(tokens, dtype=np.int32)), (".pos", np.asarray(positions, dtype=np.int32)),
                         (".st", cols.view(torch.int16).cpu().numpy()), (".tki", torch.cat(ids).cpu().numpy()),
                         (".tkl", torch.cat(logp).view(torch.int16).cpu().numpy())):
            with open(str(base) + ext, "ab") as f:
                arr.tofile(f)
        self.rows[sid] = self.rows.get(sid, 0) + len(tokens)

    def finish(self, sid: int, **meta) -> None:
        rows = self.rows.pop(sid, 0)
        if rows:
            info = {"rows": rows, "taps": list(self.taps), "k": self.k, "dims": self.dims, **meta}
            Path(str(self._files(sid)) + ".json").write_text(json.dumps(info))
