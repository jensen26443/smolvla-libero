"""One control action per tick, single-flight requests and commitment locks."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .schema import (
    ActionPlan,
    CommittedPrefix,
    InferenceRequest,
    InferenceResult,
    MethodConfig,
    ObservationSnapshot,
    digest_arrays,
    frozen_copy,
    neutral,
)
from .timing import EventLog


class Executor:
    def __init__(self, config: MethodConfig, *, generation: int, seed: int, version: str,
                 attempt_id: str | None = None, directory: Path | None = None) -> None:
        self.config, self.generation, self.seed, self.version = config, generation, seed, version
        self.active: ActionPlan | None = None
        self.previous: ActionPlan | None = None
        self.pending: InferenceRequest | None = None
        self.waiting: tuple[InferenceResult, int] | None = None
        self.last_request_tick = -config.period
        self.ordinal = 0
        self.gripper = -1.0
        self.urgent = False
        self.attempt_id = attempt_id or f"attempt-{generation:06d}"
        self.events = EventLog(attempt_id=self.attempt_id, generation=generation, directory=directory)

    def future_prefix(self, tick: int, length: int) -> CommittedPrefix:
        actions, missing = [], []
        grip = self.gripper
        for offset in range(length):
            action, _ = self.active.at(tick + offset) if self.active else (None, -1)
            if action is None:
                action = neutral(grip)
                missing.append(offset)
            elif self.config.method == "C_BLEND" and self.config.blend:
                transition = tick + offset - self.active.activation_tick
                if 0 <= transition < self.config.blend and self.active.start_index > 0:
                    old, _ = self.previous.at(tick + offset) if self.previous else (None, -1)
                    old = neutral(grip) if old is None else old
                    weight = (transition + 1) / self.config.blend
                    action[:3] = (1 - weight) * old[:3] + weight * action[:3]
            grip = float(action[-1])
            actions.append(action)
        array = frozen_copy(np.asarray(actions, dtype=np.float64).reshape(length, 7))
        return CommittedPrefix(tick, array, tuple(missing), digest_arrays(array))

    def submit(self, snapshot: ObservationSnapshot, estimate: int) -> InferenceRequest | None:
        tick = snapshot.tick
        if self.pending is not None:
            return None
        if not self.urgent and tick - self.last_request_tick < self.config.period:
            return None
        length = min(self.config.horizon, estimate + self.config.margin) if (
            self.config.method.startswith("C_") and estimate > 0) else 0
        prefix = self.future_prefix(tick, length)
        request = InferenceRequest(self.generation, self.ordinal, snapshot, tick, prefix,
                                   self.seed, time.monotonic(), self.attempt_id)
        self.ordinal += 1
        self.last_request_tick = tick
        self.pending = request
        self.urgent = False
        self.events.append({"event": "submit", "tick": tick, "request_id": request.request_id,
                            "monotonic_ns": request.submit_monotonic_ns,
                            "t_obs": snapshot.tick, "t_req": tick, "prefix_length": length,
                            "prefix_end": prefix.end_tick, "prefix_sha256": prefix.digest,
                            "neutral_prefix_positions": list(prefix.neutral_positions),
                            "observation_sha256": snapshot.digest, "estimate": estimate})
        return request

    def receive(self, result: InferenceResult, tick: int) -> bool:
        request = result.request
        if result.received_monotonic_ns is None:
            result.received_monotonic_ns = time.monotonic_ns()
        self.events.append({"event": "worker_complete", "tick": None, "request_tick": request.request_tick,
                            "request_id": request.request_id, "generation": request.generation,
                            "attempt_id": request.attempt_id or self.attempt_id,
                            "monotonic_ns": result.worker_completed_monotonic_ns, "error": result.error})
        self.events.append({"event": "control_receive", "tick": tick, "request_id": request.request_id,
                            "generation": request.generation, "monotonic_ns": result.received_monotonic_ns,
                            "attempt_id": request.attempt_id or self.attempt_id, "error": result.error})
        if (request.generation != self.generation or self.pending is None
                or request.request_id != self.pending.request_id):
            self.events.append({"event": "stale", "tick": tick, "request_id": request.request_id})
            return False
        if self.waiting is not None:
            raise RuntimeError("duplicate completion for an in-flight request")
        if (request.snapshot.digest != self.pending.snapshot.digest
                or request.prefix.digest != self.pending.prefix.digest):
            raise ValueError("worker result provenance differs from the submitted snapshot/prefix")
        if result.error:
            raise RuntimeError(f"worker inference: {result.error}")
        actions = np.asarray(result.actions)
        if actions.shape != (self.config.horizon, 7) or not np.isfinite(actions).all():
            raise ValueError("invalid model action chunk")
        if np.any(np.abs(actions) > 1.0000001):
            raise ValueError("model action outside audited bounds")
        self.events.append({"event": "ready", "tick": tick, "t_ready": tick,
                            "request_id": request.request_id, "ready_wall": result.ready_wall,
                            "stages": result.stages, "sampler": result.sampler})
        self.waiting = (result, tick)
        return True

    def advance(self, tick: int) -> None:
        if self.waiting is None:
            return
        result, ready = self.waiting
        request = result.request
        candidate = self.config.method.startswith("C_")
        end = request.prefix.end_tick
        if candidate and tick < end:
            return
        age = tick - request.snapshot.tick
        late = candidate and tick > end
        index = len(request.prefix.actions) if candidate else (age if self.config.method == "B_SHIFT" else 0)
        if late or index >= self.config.horizon:
            self.events.append({"event": "late_discard" if late else "expired_discard", "tick": tick,
                                "request_id": request.request_id, "age": age})
            if late:
                self.active = None
            self.urgent = True
        else:
            self.previous = self.active
            self.active = ActionPlan(request.request_id, request.snapshot.tick, ready, tick,
                                     index, frozen_copy(result.actions), self.version)
            self.events.append({"event": "activate", "tick": tick, "t_act": tick,
                                "request_id": request.request_id, "start_index": index,
                                "age": age, "prefix_violation": 0})
        self.pending, self.waiting = None, None

    def action(self, tick: int) -> tuple[np.ndarray, dict]:
        prefix = self.pending.prefix if self.pending else None
        if prefix is not None and prefix.obs_tick <= tick < prefix.end_tick:
            index = tick - prefix.obs_tick
            action = prefix.actions[index].copy()
            meta = {"source": "committed", "request_id": self.pending.request_id,
                    "index": index, "age": tick - prefix.obs_tick,
                    "neutral": index in prefix.neutral_positions, "prefix_violation": 0.0}
        else:
            awaiting_late = prefix is not None and tick >= prefix.end_tick and len(prefix.actions) > 0
            action, index = self.active.at(tick) if self.active and not awaiting_late else (None, -1)
            meta = {"source": "plan" if action is not None else "neutral",
                    "request_id": self.active.request_id if self.active else None,
                    "index": index, "age": tick - self.active.obs_tick if self.active else None,
                    "neutral": action is None, "prefix_violation": None}
            if action is None:
                action = neutral(self.gripper)
            elif self.config.method == "C_BLEND" and self.config.blend:
                offset = tick - self.active.activation_tick
                if offset < self.config.blend and self.active.start_index > 0:
                    old, _ = self.previous.at(tick) if self.previous else (None, -1)
                    old = neutral(self.gripper) if old is None else old
                    weight = (offset + 1) / self.config.blend
                    action[:3] = (1 - weight) * old[:3] + weight * action[:3]
                    meta["blend_weight"] = weight
        self.gripper = float(action[-1])
        return action, meta
