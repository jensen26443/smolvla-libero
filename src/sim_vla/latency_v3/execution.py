"""Reuse the audited commitment executor with explicit PLAN-05 coverage gates."""
from dataclasses import replace

from ..latency_v2.alignment import Executor
from ..latency_v2.schema import MethodConfig, frozen_copy
from .schema import CoverageInfeasible, ResearchRequest


class ResearchExecutor(Executor):
    def __init__(self, config, **kwargs):
        self.research_config = config
        mapped = "C_PREFIX" if config.prefix_method else config.method
        super().__init__(MethodConfig(mapped, config.period, margin=config.margin), **kwargs)

    def remaining(self, tick):
        if self.active is None:
            return 0
        return max(0, len(self.active.actions) - self.active.start_index
                   - (tick - self.active.activation_tick))

    def due(self, tick, estimate):
        if self.pending is not None:
            return False
        cfg = self.research_config
        period = cfg.period
        impossible = False
        if cfg.prefix_method and cfg.adaptive and estimate > 0:
            c = estimate + cfg.margin
            period = max(1, min(period, cfg.horizon-c))
            impossible = 2*c > cfg.horizon
        self.config = replace(self.config, period=period)
        due = self.urgent or tick-self.last_request_tick >= period
        # An existing valid native block may run until a new request is due.
        # Do not reject the startup's first control tick for a future request.
        if due and impossible:
            self.events.append({"event": "coverage_infeasible", "tick": tick,
                                "remaining": self.remaining(tick), "prefix_length": c,
                                "estimate": estimate, "effective_period": period,
                                "reason": "two_commitments_exceed_horizon"})
            raise CoverageInfeasible(f"2c={2*c}>H={cfg.horizon}")
        return due

    def submit(self, snapshot, estimate):
        cfg = self.research_config
        length = estimate + cfg.margin if cfg.prefix_method and estimate > 0 else 0
        remaining = self.remaining(snapshot.tick)
        if length and not cfg.legacy and (max(self.config.period, length)+length > cfg.horizon or remaining < length):
            self.events.append({"event": "coverage_infeasible", "tick": snapshot.tick,
                "remaining": remaining, "prefix_length": length, "estimate": estimate,
                "effective_period": self.config.period})
            raise CoverageInfeasible(f"remaining={remaining}, c={length}, K={self.config.period}")
        original = super().submit(snapshot, estimate)
        if original is None:
            return None
        guide = None
        if cfg.method == "RTC_GUIDED" and length:
            # Only actual old actions; the official sampler handles action padding.
            guide = self.future_prefix(snapshot.tick, min(remaining, length+3))
            if guide.neutral_positions:
                raise CoverageInfeasible("RTC guide contains unobserved future actions")
            guide = frozen_copy(guide.actions)
        request = ResearchRequest(**vars(original), sampler_mode=cfg.method,
                                  guide_actions=guide, remaining_old=remaining)
        self.pending = request
        self.events.append({"event": "coverage", "tick": snapshot.tick,
            "request_id": request.request_id, "remaining": remaining, "prefix_length": length,
            "effective_period": self.config.period, "legacy": cfg.legacy,
            "guide_length": 0 if guide is None else len(guide)})
        return request
