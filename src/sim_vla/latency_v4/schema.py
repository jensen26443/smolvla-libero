"""Resolved experiment parameters and immutable worker inputs."""
import math
from dataclasses import asdict, dataclass

import numpy as np

from ..latency_v2.schema import InferenceRequest

METHODS = ("RAW", "SHIFT", "RTC_STRICT", "RTC_CATCHUP")
SAMPLING_STEPS = (10, 7, 5)
ESTIMATOR = {"window": 10, "percentile": 95, "tick_seconds": .05}
CONTROL = {"tick_seconds": .05, "cycle_limit_seconds": .055, "required_cycle_fraction": .95,
           "max_ticks": 300, "fallback_limit_ticks": 60}


@dataclass(frozen=True)
class MethodConfig:
    method: str
    sampling_steps: int
    period: int
    initial_endpoint_p95_s: float
    horizon: int = 50
    margin: int = 2
    late_policy: str = "strict"
    max_late_ticks: int = 3
    guidance_weight: float = 10.
    guidance_schedule: str = "EXP"
    soft_overlap: int = 3
    estimator_window: int = 10
    estimator_percentile: int = 95
    tick_seconds: float = .05
    cycle_limit_seconds: float = .055
    required_cycle_fraction: float = .95
    max_ticks: int = 300
    fallback_limit_ticks: int = 60
    startup: str = "stationary_native_prefill"
    observation_mode: str = "on_request"

    def __post_init__(self):
        if self.method not in (*METHODS, "HARD") or self.sampling_steps not in SAMPLING_STEPS:
            raise ValueError("unsupported PLAN-06 method or sampling steps")
        if self.period not in (10, 25, 50) or self.horizon != 50 or self.margin != 2:
            raise ValueError("unsupported horizon, period or commitment margin")
        if (not math.isfinite(self.initial_endpoint_p95_s) or self.initial_endpoint_p95_s <= 0
                or isinstance(self.initial_endpoint_p95_s, bool)):
            raise ValueError("initial endpoint p95 must be finite and positive")
        expected_policy = "catchup" if self.method == "RTC_CATCHUP" else "strict"
        if self.late_policy != expected_policy or self.max_late_ticks != 3:
            raise ValueError("late policy must match the method, with a fixed three-tick bound")
        if (self.guidance_weight, self.guidance_schedule, self.soft_overlap) != (10., "EXP", 3):
            raise ValueError("RTC guidance differs from PLAN-06")
        if {"window": self.estimator_window, "percentile": self.estimator_percentile,
                "tick_seconds": self.tick_seconds} != ESTIMATOR:
            raise ValueError("delay estimator differs from PLAN-06")
        if {key: getattr(self, key) for key in CONTROL} != CONTROL:
            raise ValueError("control protocol differs from PLAN-06")
        if self.startup != "stationary_native_prefill" or self.observation_mode != "on_request":
            raise ValueError("startup or observation protocol differs from PLAN-06")

    @property
    def prefix_method(self):
        return self.method in ("RTC_STRICT", "RTC_CATCHUP", "HARD")

    def resolved(self):
        return asdict(self)

    @classmethod
    def from_resolved(cls, value):
        if set(value) != set(cls.__dataclass_fields__):
            raise ValueError("effective configuration is incomplete; defaults cannot repair saved trials")
        return cls(**value)


@dataclass(frozen=True)
class Request(InferenceRequest):
    sampler_mode: str = "RAW"
    sampling_steps: int = 10
    guide_actions: np.ndarray | None = None
    soft_overlap: int = 3
    remaining_old: int = 0
    effective_period: int = 50
    observation_prepare_s: float = 0.


class CoverageInfeasible(RuntimeError):
    """A scored method limitation, never a reason to rerun a formal score."""
