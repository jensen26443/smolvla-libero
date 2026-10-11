"""Instance-only instrumentation; exclusive stage times avoid double counting."""
from __future__ import annotations

import time
from collections import defaultdict
from functools import wraps


class EnvTiming:
    def __init__(self):
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)
        self.stack = []

    def wrap(self, owner, attribute: str, stage: str):
        original = getattr(owner, attribute)
        if getattr(original, "_latency_timing", None) is self:
            return

        @wraps(original)
        def measured(*args, **kwargs):
            frame = [time.monotonic_ns(), 0]
            self.stack.append(frame)
            try:
                return original(*args, **kwargs)
            finally:
                elapsed = time.monotonic_ns()-frame[0]
                self.stack.pop()
                self.seconds[stage] += (elapsed-frame[1])/1e9
                self.calls[stage] += 1
                if self.stack:
                    self.stack[-1][1] += elapsed

        measured._latency_timing = self
        setattr(owner, attribute, measured)

    def take(self, total: float) -> dict:
        result = {"total_s": total, "exclusive_s": dict(self.seconds), "calls": dict(self.calls),
                  "unattributed_s": total-sum(self.seconds.values())}
        self.seconds.clear()
        self.calls.clear()
        return result
