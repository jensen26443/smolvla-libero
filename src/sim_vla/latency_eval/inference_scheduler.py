"""Single-in-flight inference scheduler with discrete-event result delay.

The design is a *discrete-event latency simulation*, not a real asynchronous
GPU service: the wall-clock forward pass runs immediately at request time on a
frozen input snapshot, but its result is only visible to the executor from
``completion_tick`` onward. Ticks that pass while the wall clock computes are
still simulated one by one by the runner, so stale actions and physics keep
evolving exactly as they would under real delay.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from .contracts import ActionPlan, ActionSpec, InferenceRequest, ObservationPacket


class LatencySchedule(Protocol):
    """Maps a request tick to the inference-result delay in ticks."""

    def inference_delay_ticks(self, tick: int) -> int: ...


class FixedDelay:
    def __init__(self, ticks: int) -> None:
        if ticks < 0:
            raise ValueError(f"inference delay must be >= 0 ticks, got {ticks}")
        self.ticks = int(ticks)

    def inference_delay_ticks(self, tick: int) -> int:
        return self.ticks

    def describe(self) -> dict[str, Any]:
        return {"kind": "fixed", "ticks": self.ticks}


class JitteredDelay:
    """Tick-indexed multiplicative jitter over a base delay.

    The multiplier trajectory is generated once per experiment from an external
    seeded RNG and saved to disk; every method reads the *same tick-indexed*
    curve, so methods with different request counts face the same exogenous
    latency process at the same simulation times (pairing by request ordinal
    would not guarantee that and is deliberately not used).
    """

    def __init__(self, base_ticks: int, multipliers: Any) -> None:
        self.base_ticks = int(base_ticks)
        self._multipliers = np.asarray(multipliers, dtype=np.float64)
        if self._multipliers.ndim != 1:
            raise ValueError("jitter multipliers must be a 1-D trajectory")

    def inference_delay_ticks(self, tick: int) -> int:
        index = min(max(tick, 0), len(self._multipliers) - 1)
        value = float(self._multipliers[index]) * self.base_ticks
        return max(0, int(np.ceil(value)))

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "jittered",
            "base_ticks": self.base_ticks,
            "trajectory_length": len(self._multipliers),
            "multiplier_min": float(self._multipliers.min()),
            "multiplier_max": float(self._multipliers.max()),
        }


class InferenceFn(Protocol):
    """Frozen policy inference on a snapshot: packet -> (H, dim) actions."""

    def __call__(
        self,
        packet: ObservationPacket,
        request_id: int,
        request_tick: int,
    ) -> Any:  # returns np.ndarray (H, dim) plus may record timing
        ...


class InferenceScheduler:
    """Owns request ids, the single in-flight slot, and delayed completions."""

    def __init__(
        self,
        schedule: LatencySchedule,
        inference_fn: InferenceFn,
        action_spec: ActionSpec,
        *,
        checkpoint: str = "",
        config_hash: str = "",
        completion_hook: Callable[[ActionPlan], None] | None = None,
    ) -> None:
        self.schedule = schedule
        self.inference_fn = inference_fn
        self.action_spec = action_spec
        self.checkpoint = checkpoint
        self.config_hash = config_hash
        self.completion_hook = completion_hook
        self._next_request_id = 0
        self._in_flight: InferenceRequest | None = None
        self._completed_pending_release: list[ActionPlan] = []

    @property
    def in_flight(self) -> InferenceRequest | None:
        return self._in_flight

    @property
    def in_flight_request(self) -> bool:
        return self._in_flight is not None

    def submit(
        self,
        tick: int,
        packet: ObservationPacket,
        reason: str,
    ) -> InferenceRequest:
        if self._in_flight is not None:
            raise RuntimeError("cannot submit while a request is in flight")
        request = InferenceRequest(
            request_id=self._next_request_id,
            input_packet_id=packet.packet_id,
            input_capture_tick=packet.capture_tick,
            request_tick=tick,
            completion_tick=tick + self.schedule.inference_delay_ticks(tick),
            checkpoint=self.checkpoint,
            config_hash=self.config_hash,
        )
        self._next_request_id += 1
        # The request stays "in flight" (blocking new submissions) until its
        # result is released at completion_tick, even though the wall-clock
        # forward pass below runs synchronously right now.
        self._in_flight = request
        raw_actions = self.inference_fn(packet, request.request_id, request.request_tick)
        actions = np.asarray(raw_actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != self.action_spec.dim:
            raise ValueError(
                f"inference returned shape {actions.shape}, expected (H, {self.action_spec.dim})"
            )
        actions = self.action_spec.clip(actions)
        # Sealed until release_due(tick >= completion_tick) hands it over.
        self._completed_pending_release.append(
            ActionPlan(
                plan_id=request.request_id,
                request=request,
                actions=actions,
                available_tick=request.completion_tick,
            )
        )
        return request

    def release_due(self, tick: int) -> list[ActionPlan]:
        due = [plan for plan in self._completed_pending_release if plan.available_tick <= tick]
        if due:
            self._completed_pending_release = [
                plan for plan in self._completed_pending_release if plan.available_tick > tick
            ]
            if self._in_flight is not None and any(
                plan.request.request_id == self._in_flight.request_id for plan in due
            ):
                self._in_flight = None
        return due

    def reset(self) -> None:
        self._in_flight = None
        self._completed_pending_release.clear()
