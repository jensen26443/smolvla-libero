"""Time, observation, request, and action-plan contracts for latency evaluation.

All timestamps in this package are integer *ticks* of the control clock. One
tick is one environment control step (LIBERO Goal: 20 Hz, 50 ms). Physics
substeps stay inside robosuite; this layer never interpolates or substeps.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

# LIBERO Goal runs at control_freq=20 with a 300-step episode cap. These are
# recorded once here so every module shares the same clock definition.
CONTROL_HZ = 20.0
TICK_SECONDS = 1.0 / CONTROL_HZ
DEFAULT_MAX_TICKS = 300

# The officially sanctioned no-op action (lerobot.envs.libero.get_libero_dummy_action):
# zero end-effector delta with a legal gripper command. Used verbatim as the
# fallback action whenever no executable plan exists. Delta semantics forbid
# repeating the previous delta, so "hold still" is this zero action, not the
# last commanded increment.
FALLBACK_ACTION = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)


@dataclass(frozen=True)
class ActionSpec:
    """Semantic description of the action space shared by every method.

    All methods share the same de-normalization and clipping; this record makes
    the semantics auditable instead of implicit in code.
    """

    output_type: str = "relative_ee_pose_delta_with_gripper"
    dim: int = 7
    horizon: int = 50  # H: number of actions in one model chunk
    low: float = -1.0
    high: float = 1.0
    frame: str = "robot_base_ee_delta"  # OSC_POSE relative deltas
    gripper_encoding: str = "last_dim_signed_command"
    fallback: tuple[float, ...] = FALLBACK_ACTION
    per_step_units: str = "normalized_delta_at_20hz"

    def clip(self, actions: np.ndarray) -> np.ndarray:
        return np.clip(actions, self.low, self.high)


class SwitchRule(str, Enum):
    """How a newly available plan becomes the active plan."""

    IMMEDIATE = "immediate"  # B1/M/R0: activate at the current executable boundary
    SEGMENT_BOUNDARY = "segment_boundary"  # B2: wait for the next multiple of `segment_ticks`


@dataclass(frozen=True)
class ObservationPacket:
    """One immutable sensing event.

    ``images`` and ``robot_state`` are deep-copied at construction so a later
    mutation of the environment's observation cache can never retroactively
    change an inference input (the input-snapshot guarantee). ``robot_state``
    keeps the nested eef/gripper structure verbatim so the official
    LiberoProcessorStep can flatten it exactly as in the unmodified eval path.
    """

    packet_id: int
    capture_tick: int
    arrive_tick: int
    images: dict[str, np.ndarray]
    robot_state: dict[str, Any] | None
    instruction: str
    modality_capture_ticks: dict[str, int]

    @property
    def sensor_delay_ticks(self) -> int:
        return self.arrive_tick - self.capture_tick

    def capture_age_ticks(self, tick: int) -> int:
        """Age of the underlying scene when observed at ``tick`` (>=1 once captured)."""
        return tick - self.capture_tick


@dataclass(frozen=True)
class InferenceRequest:
    request_id: int
    input_packet_id: int
    input_capture_tick: int
    request_tick: int
    completion_tick: int
    checkpoint: str
    config_hash: str

    @property
    def inference_delay_ticks(self) -> int:
        return self.completion_tick - self.request_tick


@dataclass
class ActionPlan:
    """A chunk of actions produced by one inference request.

    ``available_tick`` is the first tick at which the executor may even look at
    the plan; ``activated_tick`` records when it actually became active.
    """

    plan_id: int
    request: InferenceRequest
    actions: np.ndarray  # (H, dim) post-processed and clipped
    available_tick: int
    activated_tick: int | None = None
    consumption_index: int = 0

    @property
    def exhausted(self) -> bool:
        return self.consumption_index >= len(self.actions)

    def next_action(self) -> np.ndarray:
        action = self.actions[self.consumption_index]
        self.consumption_index += 1
        return action

    def plan_age_ticks(self, tick: int) -> int:
        """Age of the underlying observation for actions executed at ``tick``."""
        return tick - self.request.input_capture_tick


class ActionSource(str, Enum):
    PLAN = "plan"
    FALLBACK_NO_PLAN = "fallback_no_plan"
    FALLBACK_EXHAUSTED = "fallback_exhausted"
    FALLBACK_INVALID = "fallback_invalid_action"


@dataclass
class TriggerDecision:
    fire: bool
    reason: str  # "image_diff" | "max_interval" | "period" | "every_step"
    detail: dict[str, float] = field(default_factory=dict)


@dataclass
class StepEvent:
    """One control tick's audit trail (compact; arrays live in .npz sidecars)."""

    tick: int
    arrived_packet_ids: list[int]
    latest_packet_id: int | None
    request_submitted: bool
    request_id: int | None
    request_reason: str | None
    trigger_suppressed_reason: str | None
    trigger_detail: dict[str, float]
    completed_request_ids: list[int]
    activated_plan_id: int | None
    action_source: str
    action_summary: list[float]
    fallback_reason: str | None
    active_plan_id: int | None
    active_plan_age_ticks: int | None
    input_obs_age_ticks: int | None
    in_flight: bool
    wall_time_s: float

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "arrived_packet_ids": self.arrived_packet_ids,
            "latest_packet_id": self.latest_packet_id,
            "request_submitted": self.request_submitted,
            "request_id": self.request_id,
            "request_reason": self.request_reason,
            "trigger_suppressed_reason": self.trigger_suppressed_reason,
            "trigger_detail": {k: round(v, 4) for k, v in self.trigger_detail.items()},
            "completed_request_ids": self.completed_request_ids,
            "activated_plan_id": self.activated_plan_id,
            "action_source": self.action_source,
            "action_summary": [round(float(v), 4) for v in self.action_summary],
            "fallback_reason": self.fallback_reason,
            "active_plan_id": self.active_plan_id,
            "active_plan_age_ticks": self.active_plan_age_ticks,
            "input_obs_age_ticks": self.input_obs_age_ticks,
            "in_flight": self.in_flight,
            "wall_time_s": round(self.wall_time_s, 6),
        }


@dataclass
class PolicyView:
    """Everything a trigger or policy may condition on.

    Deliberately excludes evaluator ground truth (object poses, contacts,
    success flags). Only observations that have *arrived* by the current tick
    are visible, plus the robot's own sent commands and the fixed instruction.
    """

    instruction: str
    latest_packet: ObservationPacket | None = None
    arrived_count: int = 0
    sent_commands: deque[tuple[int, np.ndarray]] = field(default_factory=lambda: deque(maxlen=64))

    def on_packet_arrived(self, packet: ObservationPacket) -> None:
        self.latest_packet = packet
        self.arrived_count += 1

    def on_command_sent(self, tick: int, action: np.ndarray) -> None:
        self.sent_commands.append((tick, np.array(action, copy=True)))


@dataclass
class EpisodeOutcome:
    """Evaluator-side record of one episode (never fed back into control)."""

    success: bool
    termination_reason: str  # "success" | "env_done" | "timeout" | "invalid_action" | "error"
    steps: int
    first_success_step: int | None
    n_requests: int
    inference_seconds_total: float
    wall_seconds: float
    error: str | None = None


def monotonic_wall_s() -> float:
    return time.monotonic()


def new_run_id() -> str:
    return f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
