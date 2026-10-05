"""The ``Scheduler``'s decoder for Kolibri 1: a prompt chunk a round, then every decoding stream's window together."""
# Windows copy what followed the context's last 8 tokens before; rows are row-invariant, so drafted equals serial.
# A free slot keeps its prompt rows (prompt kernels' bits); a prompt extending them resumes there, equal to fresh.
# With a learned drafter, a stream whose copies find nothing drafts its chain instead; the target verifies either.

from __future__ import annotations

import time

import numpy as np
import torch

from tensorfold.cuda.memory_gate import NoRoom
from tensorfold.cuda.sampling import sample_rows
from tensorfold.cuda.streams import Stream, accept, next_fill
from tensorfold.families.qwen3_5.cuda.decode import CopyIndex, next_copy_rows

from .drafter import WINDOW
from .forward import PROMPT_CHUNK, Chain, Model

STEP = 1024              # prompt rows a round while other streams decode
MAX_ROWS = 32            # a copy window's rows: each wrong one reads its own experts
MIN_MATCH = 8            # context tokens a copy must repeat before it is proposed


class Decoder:
    def __init__(self, model: Model, eos: tuple[int, ...], recorder=None, drafter=None,
                 draft_streams: int = 64) -> None:
        self.model, self.eos, self.recorder = model, tuple(eos), recorder
        self.drafter, self.draft_streams = drafter, draft_streams
        self.learned: dict[int, list[int]] = {}    # the drafter's chain for each stream's next round
        self.learned_rows = self.learned_kept = 0
        if drafter is not None:                    # its taps come from every forward's kept layer states
            if model.record_taps and tuple(model.record_taps) != drafter.taps:
                raise ValueError(f"the recorder's layers {model.record_taps} differ from the drafter's {drafter.taps}")
            model.record_taps = drafter.taps
        self.free = list(range(model.slots))
        self.streams: dict[int, Stream] = {}      # decoding
        self.filling: list[Stream] = []           # admitted, prompts still prefilling (oldest first)
        self.slot: dict[int, int] = {}
        self.done_at: dict[int, int] = {}         # prompt rows prefilled
        self.copies: dict[int, CopyIndex] = {}
        self.context: dict[int, list[int]] = {}   # prompt and reply, for the copy index
        self.width: dict[int, int] = {}
        self.written: dict[int, int] = {}         # positions a stream has written its slot through
        self.kept: dict[int, tuple[np.ndarray, int, int]] = {}   # free slot -> (prompt rows, written, last use)
        self.tick = 0
        self.next_id = 0

    def live(self) -> int:
        return len(self.streams) + len(self.filling)

    def admit(self, s: Stream) -> None:
        if s.constraint is not None:
            raise ValueError("Kolibri 1 on CUDA does not take response_format grammars yet")
        room = self.model.context - len(s.prompt)
        if len(s.prompt) < 1 or room < 1:
            raise ValueError(f"a prompt of {len(s.prompt)} tokens leaves no room in the {self.model.context}-token "
                             "context (--context)")
        if not self.free:
            raise NoRoom("every stream slot is busy")
        s.count = min(s.count, room)
        s.sid = self.next_id
        self.next_id += 1
        slot, start = self._resume(s)
        self.free.remove(slot)
        self.kept.pop(slot, None)
        self.slot[s.sid] = slot
        self.done_at[s.sid] = self.written[s.sid] = s.cached = start
        self.filling.append(s)

    def _resume(self, s: Stream) -> tuple[int, int]:
        """The free slot whose kept prompt rows the prompt extends furthest (within its ring), else the oldest."""

        best, start = None, 0
        prompt = np.asarray(s.prompt, dtype=np.int64)
        guard = self.model.ring - self.model.cfg.window          # rows written past a resume point that keep its window
        for slot in self.free if s.draft else ():               # "draft": false is the serial reference: never resumed
            if slot not in self.kept:
                continue
            ids, written, _ = self.kept[slot]
            n = min(len(ids), len(prompt) - 1)                   # a resumed prompt still prefills its last row
            miss = np.flatnonzero(ids[:n] != prompt[:n])
            lcp = int(miss[0]) if len(miss) else n
            if lcp > start and written - lcp <= guard:
                best, start = slot, lcp
        if best is None:
            best = min(self.free, key=lambda x: self.kept[x][2] if x in self.kept else -1)
        return best, start

    def _ends(self, s: Stream) -> tuple[int, ...]:
        return self.eos if s.stop_eos else ()

    def _fill(self) -> list[Stream]:
        s = next_fill(self.filling)
        a = self.done_at[s.sid]
        b = min(len(s.prompt), a + (STEP if self.streams else PROMPT_CHUNK))
        t0 = time.perf_counter()
        try:
            logits = self.model.forward([Chain(self.slot[s.sid], a, s.prompt[a:b])], prompt=True)
            first = sample_rows(logits, [len(s.prompt)], s.sampling)[0] if b == len(s.prompt) else None
        except Exception as exc:                 # noqa: BLE001  (this request fails, the others go on)
            self.filling.remove(s)
            s.error, s.done = exc, True
            return [s]
        self.done_at[s.sid] = b
        self.written[s.sid] = max(self.written[s.sid], b)
        if self.recorder is not None:
            final, states = self.model.last
            self.recorder.add(s.sid, range(a, b), s.prompt[a:b], states, final, self.model.w.head)
        if self.drafter is not None and s.draft:            # its last rows (the trained window) enter the drafter
            lo = max(a, len(s.prompt) - 1 - WINDOW)
            nxt = list(s.prompt[lo + 1:b + 1]) + ([first] if first is not None else [])   # each row's next token
            if nxt:
                _, states = self.model.last
                rows = slice(lo - a, lo - a + len(nxt))
                self.drafter.add(s.sid, list(range(lo, lo + len(nxt))), nxt, [t[rows] for t in states])
        s.prefill_s += time.perf_counter() - t0
        if first is None:
            return []
        self.filling.remove(s)
        self.streams[s.sid] = s
        s.started = time.perf_counter()
        self.context[s.sid] = list(s.prompt)
        self.copies[s.sid] = CopyIndex(MIN_MATCH)
        self.width[s.sid] = 16                       # a copy window's rows, doubled after a whole copy
        s.take([first], self._ends(s))
        self.context[s.sid].append(first)
        if self.drafter is not None and s.draft:
            self.learned[s.sid] = self.drafter.chain(s.sid)
        return [s] if s.done else []

    def round(self) -> list[Stream]:
        """A prompt chunk for the next queued prompt, then a row of each decoding stream; returns the finished."""

        done = self._fill() if self.filling else []
        live = [s for s in self.streams.values() if not s.done]
        if not live:
            return done
        chains, starts, copied = [], [], {}
        learn = self.drafter is not None and len(live) <= self.draft_streams
        for s in live:
            p = len(s.prompt) + len(s.out) - 1             # the pending token's position
            room = min(s.count - len(s.out), self.model.context - p) - 1
            drafts = []
            if s.draft and room > 0:
                most = min(room, self.width[s.sid] - 1)
                drafts = self.copies[s.sid].propose(self.context[s.sid], most)[:most]
                copied[s.sid] = bool(drafts)
                if not drafts and learn:
                    drafts = self.learned.get(s.sid, [])[:room]
                    self.learned_rows += len(drafts)
            starts.append(sum(len(c.tokens) for c in chains))
            chains.append(Chain(self.slot[s.sid], p, [s.out[-1], *drafts]))
            self.written[s.sid] = max(self.written[s.sid], p + 1 + len(drafts))
        total = sum(len(c.tokens) for c in chains)
        logits = self.model.forward(chains, prompt=False, rows=range(total))
        for s, c, a in zip(live, chains, starts):
            win = list(c.tokens)
            sampled = sample_rows(logits[a:a + len(win)], [c.p0 + 1 + i for i in range(len(win))], s.sampling)
            path, last = accept(win, list(range(-1, len(win) - 1)), sampled, s.count - len(s.out), self._ends(s))
            if self.recorder is not None:                   # the kept rows only: a rejected draft's state is not one
                final, states = self.model.last
                keep = torch.tensor([a + r for r in path], device=final.device)
                self.recorder.add(s.sid, [c.p0 + r for r in path], [win[r] for r in path],
                                  [t[keep] for t in states], final[keep], self.model.w.head, kinds=1)
            new = [win[r] for r in path[1:]] + [last]
            s.counted(len(win))
            if len(win) > 1 and copied.get(s.sid):
                self.width[s.sid] = next_copy_rows(len(win), len(path) == len(win), 1, MAX_ROWS)
            elif len(win) > 1:
                self.learned_kept += len(path) - 1
            s.take(new, self._ends(s))
            self.context[s.sid].extend(new)
            if self.drafter is not None and s.draft:       # the kept rows enter the drafter, then the next chain
                _, states = self.model.last
                keep = torch.tensor([a + r for r in path], device=states[0].device)
                self.drafter.add(s.sid, [c.p0 + r for r in path], new, [t[keep] for t in states])
                if not s.done and learn:
                    self.learned[s.sid] = self.drafter.chain(s.sid)
        return done + [s for s in live if s.done]

    def finish(self, done: list[Stream]) -> None:
        for s in done:
            if self.recorder is not None:
                self.recorder.finish(s.sid, prompt=len(s.prompt), cached=s.cached, reply=len(s.out))
            self.streams.pop(s.sid, None)
            if s in self.filling:
                self.filling.remove(s)
            slot = self.slot.pop(s.sid, None)
            rows = self.done_at.pop(s.sid, 0)
            if slot is not None:
                self.tick += 1
                ids = np.asarray(s.prompt[:rows], dtype=np.int64)
                self.kept[slot] = (ids, self.written.pop(s.sid, rows), self.tick)
                self.free.append(slot)
                self.free.sort()
            for d in (self.copies, self.context, self.width, self.learned):
                d.pop(s.sid, None)
            if self.drafter is not None:
                self.drafter.drop(s.sid)

    def drop(self) -> list[Stream]:
        gone = [s for s in self.streams.values() if not s.done] + list(self.filling)
        self.streams, self.filling = {}, []
        self.free = list(range(self.model.slots))
        self.slot.clear()
        self.done_at.clear()
        self.written.clear()
        self.kept.clear()
        self.copies.clear()
        self.context.clear()
        self.width.clear()
        self.learned.clear()
        if self.drafter is not None:
            self.drafter.cache.clear()
        return gone
