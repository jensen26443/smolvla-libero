"""Shared single-flight scheduling; only the late activation rule differs."""
from dataclasses import replace

import numpy as np

from ..latency_v2.alignment import Executor
from ..latency_v2.schema import ActionPlan, frozen_copy
from ..latency_v2.schema import MethodConfig as BaseConfig
from .schema import CoverageInfeasible, Request


class CatchupExecutor(Executor):
    def __init__(self, config, **kwargs):
        self.settings = config
        mapped = "C_PREFIX" if config.prefix_method else {"RAW": "B_RAW", "SHIFT": "B_SHIFT"}[config.method]
        super().__init__(BaseConfig(mapped, config.period, margin=config.margin), **kwargs)
        self.last_command = None
        self.fallback_by_request = {}

    def remaining(self, tick):
        if self.active is None:
            return 0
        return max(0, len(self.active.actions)-self.active.start_index-(tick-self.active.activation_tick))

    def due(self, tick, estimate):
        if self.pending is not None:
            return False
        c = estimate+self.settings.margin if self.settings.prefix_method and estimate > 0 else 0
        period = min(self.settings.period, self.settings.horizon-c) if c else self.settings.period
        due = self.urgent or tick-self.last_request_tick >= max(1, period)
        if due and 2*c > self.settings.horizon:
            self._coverage_error(tick, c, estimate, period, "two_commitments_exceed_horizon")
        self.config = replace(self.config, period=max(1, period))
        return due

    def _coverage_error(self, tick, c, estimate, period, reason):
        self.events.append({"event": "coverage_infeasible", "tick": tick, "prefix_length": c,
            "remaining_old": self.remaining(tick), "estimate": estimate,
            "effective_period": period, "reason": reason})
        raise CoverageInfeasible(f"{reason}: remaining={self.remaining(tick)}, c={c}, K={period}")

    def submit(self, snapshot, estimate, *, observation_prepare_s=0.):
        if self.pending is not None:
            return None
        c = estimate+self.settings.margin if self.settings.prefix_method and estimate > 0 else 0
        remaining = self.remaining(snapshot.tick)
        if c and (2*c > self.settings.horizon or remaining < c):
            self._coverage_error(snapshot.tick, c, estimate, self.config.period, "insufficient_old_actions")
        original = super().submit(snapshot, estimate)
        if original is None:
            return None
        guide = None
        if self.settings.method.startswith("RTC_") and c:
            prefix = self.future_prefix(snapshot.tick, min(remaining, c+self.settings.soft_overlap))
            if prefix.neutral_positions:
                raise CoverageInfeasible("guidance contains unavailable old actions")
            guide = frozen_copy(prefix.actions)
        request = Request(**vars(original), sampler_mode=self.settings.method,
            sampling_steps=self.settings.sampling_steps, guide_actions=guide,
            soft_overlap=self.settings.soft_overlap, remaining_old=remaining,
            effective_period=self.config.period, observation_prepare_s=observation_prepare_s)
        self.pending = request
        self.events.append({"event": "coverage", "tick": snapshot.tick, "request_id": request.request_id,
            "remaining_old": remaining, "prefix_length": c, "commitment_boundary": snapshot.tick+c,
            "effective_period": self.config.period, "sampling_steps": request.sampling_steps,
            "guide_length": 0 if guide is None else len(guide)})
        return request

    def receive(self, result, tick):
        if (self.waiting is not None and self.pending is not None
                and result.request.generation == self.generation
                and result.request.request_id == self.pending.request_id):
            self.events.append({"event": "duplicate_discard", "tick": tick,
                                "request_id": result.request.request_id})
            return False
        if self.pending is not None and result.request.generation == self.generation:
            pending = self.pending
            if result.request.request_id == pending.request_id:
                from ..latency_v2.schema import digest_arrays

                guide_digest = digest_arrays(result.request.guide_actions)
                if (result.request.sampling_steps != pending.sampling_steps
                        or result.request.sampler_mode != pending.sampler_mode
                        or guide_digest != digest_arrays(pending.guide_actions)):
                    raise ValueError("worker changed frozen sampling or guidance inputs")
                if result.sampler and result.sampler.get("steps") != pending.sampling_steps:
                    raise ValueError("worker sampling steps differ from the submitted configuration")
        accepted = super().receive(result, tick)
        if accepted:
            request = result.request
            self.events.append({"event": "receive_age", "tick": tick, "request_id": request.request_id,
                "commitment_boundary": request.prefix.end_tick,
                "receive_age": tick-request.snapshot.tick,
                "late_ticks_at_receive": tick-request.prefix.end_tick})
        return accepted

    def advance(self, tick):
        if self.waiting is None:
            return
        result, ready = self.waiting
        request = result.request
        candidate = self.settings.prefix_method
        boundary = request.prefix.end_tick
        if candidate and tick < boundary:
            return
        age, late = tick-request.snapshot.tick, tick-boundary
        catchup_eligible = candidate and 1 <= late <= self.settings.max_late_ticks and age < self.settings.horizon
        index = (age if late > 0 else len(request.prefix.actions)) if candidate else (
            age if self.settings.method == "SHIFT" else 0)
        rejection = None
        if (candidate and age >= self.settings.horizon) or index >= self.settings.horizon:
            rejection = "block_expired"
        elif candidate and late > 0:
            if self.settings.late_policy == "strict":
                rejection = "strict_late_discard"
            elif not catchup_eligible:
                rejection = "late_limit_exceeded"
        common = {"tick": tick, "request_id": request.request_id, "commitment_boundary": boundary,
            "receive_age": ready-request.snapshot.tick, "activation_age": age, "age": age,
            "late_ticks": max(0, late), "catchup_eligible": catchup_eligible,
            "late_fallback_ticks": self.fallback_by_request.get(request.request_id, 0)}
        if rejection:
            self.events.append({"event": "expired_discard" if rejection == "block_expired" else "late_discard",
                                **common, "reason": rejection, "catchup_accepted": False})
            if candidate:
                self.active = None
            self.urgent = True
        else:
            self.previous = self.active
            self.active = ActionPlan(request.request_id, request.snapshot.tick, ready, tick,
                                     index, frozen_copy(result.actions), self.version)
            first = self.active.actions[index]
            delta = first-self.last_command if self.last_command is not None else None
            self.events.append({"event": "activate", **common, "t_act": tick, "start_index": index,
                "prefix_violation": 0, "catchup_accepted": bool(candidate and late > 0),
                "skipped_actions": index, "remaining_new": self.settings.horizon-index,
                "switch_command_delta": None if delta is None else delta.tolist(),
                "switch_command_l2": None if delta is None else float(np.linalg.norm(delta)),
                "switch_motion_l2": None if delta is None else float(np.linalg.norm(delta[:6]))})
        self.pending, self.waiting = None, None

    def action(self, tick):
        action, meta = super().action(tick)
        if self.pending is not None and len(self.pending.prefix.actions) and tick >= self.pending.prefix.end_tick:
            request_id = self.pending.request_id
            self.fallback_by_request[request_id] = self.fallback_by_request.get(request_id, 0)+1
            meta.update(late_fallback=True, waiting_request_id=request_id)
        else:
            meta["late_fallback"] = False
        self.last_command = action.copy()
        return action, meta
