"""Explicit sampler choice, guide inputs and coverage requirements."""
from dataclasses import dataclass

import numpy as np

from ..latency_v2.schema import InferenceRequest

METHODS = ("B_RAW", "B_SHIFT", "P_HARD", "RTC_GUIDED")


@dataclass(frozen=True)
class ResearchConfig:
    method: str
    period: int
    margin: int = 2
    legacy: bool = False
    adaptive: bool = False
    horizon: int = 50

    def __post_init__(self):
        if self.method not in METHODS or self.period < 1 or self.margin < 0 or self.horizon != 50:
            raise ValueError("invalid PLAN-05 method or timing")
        if self.legacy and self.method != "P_HARD":
            raise ValueError("only the declared hard-prefix ablation permits legacy filling")

    @property
    def prefix_method(self):
        return self.method in ("P_HARD", "RTC_GUIDED")


@dataclass(frozen=True)
class ResearchRequest(InferenceRequest):
    sampler_mode: str = "B_RAW"
    guide_actions: np.ndarray | None = None
    soft_overlap: int = 3
    remaining_old: int = 0


class CoverageInfeasible(RuntimeError):
    """A scored policy limitation, not an infrastructure retry opportunity."""
