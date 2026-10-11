"""Copied inputs and explicit control-clock provenance for PLAN-04."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..latency_eval.contracts import ObservationPacket


def frozen_copy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        result = value.copy()
        result.flags.writeable = False
        return result
    if isinstance(value, dict):
        return {key: frozen_copy(item) for key, item in value.items()}
    return value


def digest_arrays(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key in sorted(item):
                digest.update(key.encode())
                visit(item[key])
        elif isinstance(item, np.ndarray):
            digest.update(str((item.shape, item.dtype.str)).encode())
            digest.update(item.tobytes())
        else:
            digest.update(str(item).encode())

    visit(value)
    return digest.hexdigest()


@dataclass(frozen=True)
class ObservationSnapshot:
    episode: str
    tick: int
    images: dict
    robot_state: dict
    language: str
    digest: str
    sampled: dict = field(default_factory=dict)
    capture_seconds: float = 0.0

    @classmethod
    def capture(cls, episode: str, tick: int, observation: dict, language: str) -> ObservationSnapshot:
        start = time.monotonic_ns()
        images = frozen_copy(observation["pixels"])
        state = frozen_copy(observation["robot_state"])
        digest = digest_arrays((digest_arrays(images), digest_arrays(state), language))
        return cls(episode, tick, images, state, language, digest,
                   frozen_copy(observation.get("sampling", {})), (time.monotonic_ns()-start)/1e9)

    def packet(self) -> ObservationPacket:
        camera_ticks = {key: self.sampled.get(key, {}).get("tick", self.tick) for key in self.images}
        if any(tick != self.tick for tick in camera_ticks.values()):
            raise ValueError("request images were not sampled at the request tick")
        return ObservationPacket(self.tick, self.tick, self.tick, self.images, self.robot_state,
                                 self.language, camera_ticks)


@dataclass(frozen=True)
class CommittedPrefix:
    obs_tick: int
    actions: np.ndarray
    neutral_positions: tuple[int, ...]
    digest: str

    @property
    def end_tick(self) -> int:
        return self.obs_tick + len(self.actions)


@dataclass(frozen=True)
class InferenceRequest:
    generation: int
    request_id: int
    snapshot: ObservationSnapshot
    request_tick: int
    prefix: CommittedPrefix
    seed: int
    submit_wall: float
    attempt_id: str = ""
    submit_monotonic_ns: int = field(default_factory=time.monotonic_ns)


@dataclass
class InferenceResult:
    request: InferenceRequest
    actions: np.ndarray | None
    ready_wall: float
    stages: dict[str, float] = field(default_factory=dict)
    sampler: dict = field(default_factory=dict)
    error: str | None = None
    worker_completed_monotonic_ns: int = field(default_factory=time.monotonic_ns)
    received_monotonic_ns: int | None = None


@dataclass
class ActionPlan:
    request_id: int
    obs_tick: int
    ready_tick: int
    activation_tick: int
    start_index: int
    actions: np.ndarray
    model_version: str

    def at(self, tick: int) -> tuple[np.ndarray | None, int]:
        index = self.start_index + tick - self.activation_tick
        return (self.actions[index].copy() if 0 <= index < len(self.actions) else None), index


def neutral(gripper: float) -> np.ndarray:
    result = np.zeros(7, dtype=np.float64)
    result[-1] = gripper
    return result


@dataclass(frozen=True)
class MethodConfig:
    method: str
    period: int
    margin: int = 0
    blend: int = 0
    horizon: int = 50

    def __post_init__(self) -> None:
        if self.method not in {"B_RAW", "B_SHIFT", "C_PREFIX", "C_BLEND"}:
            raise ValueError(self.method)
        if self.period < 1 or self.margin < 0 or self.blend < 0:
            raise ValueError("invalid scheduler parameters")
