"""The single control clock: one discrete-event tick loop per episode.

Fixed per-tick event order (PLAN-03 §5.1); changing this order changes the
semantics of every method and is not allowed after the timing tests freeze:

1. capture an observation snapshot of the state at the start of the tick and
   enqueue it with the sensor delay;
2. release every observation due this tick and update the policy view;
3. release every inference result due this tick and offer it to the executor;
4. if no request is in flight, let the trigger decide whether to request;
5. for zero-delay requests submitted this tick, release the result before
   action selection (at most one request per tick, so no infinite loop);
6. the executor selects exactly one action under the switch rule, or the
   shared fallback action when no executable plan exists;
7. execute exactly one environment control step; the evaluator records truth,
   termination, and reward — none of it visible to this tick's control.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .contracts import (
    ActionSource,
    ActionSpec,
    EpisodeOutcome,
    ObservationPacket,
    PolicyView,
    StepEvent,
    SwitchRule,
)
from .inference_scheduler import InferenceScheduler, LatencySchedule
from .plan_executor import PlanExecutor
from .sensor_queue import SensorQueue
from .triggers import Trigger


class LatencyEnv(Protocol):
    """Environment surface the runner needs (real or mock)."""

    instruction: str

    def start_episode(self, init_index: int, seed: int) -> dict[str, Any]: ...

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]: ...


class PolicyAdapter(Protocol):
    """Frozen policy surface: snapshot in, action chunk out."""

    checkpoint_label: str

    def infer(self, packet: ObservationPacket, request_id: int, request_tick: int) -> np.ndarray: ...

    def reset(self) -> None: ...


@dataclass
class RequestRecord:
    request_id: int
    request_tick: int
    input_capture_tick: int
    completion_tick: int
    reason: str
    inference_seconds: float
    stage_seconds: dict[str, float] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "request_tick": self.request_tick,
            "input_capture_tick": self.input_capture_tick,
            "completion_tick": self.completion_tick,
            "reason": self.reason,
            "inference_seconds": round(self.inference_seconds, 6),
            "stage_seconds": {k: round(v, 6) for k, v in self.stage_seconds.items()},
        }


@dataclass
class EpisodeTrace:
    outcome: EpisodeOutcome
    events: list[StepEvent]
    requests: list[RequestRecord]
    actions: np.ndarray  # (T, dim) executed actions
    success_flags: list[bool]

    @property
    def ticks(self) -> int:
        return len(self.events)


@dataclass
class EpisodeComponents:
    """Fresh per-episode components; cross-episode leakage is impossible by
    construction because nothing here survives an episode (PLAN-03 §5.3)."""

    sensor_queue: SensorQueue
    scheduler: InferenceScheduler
    executor: PlanExecutor
    trigger: Trigger
    view: PolicyView


class _TimedInference:
    """Wraps a policy adapter and records the wall-clock inference duration."""

    def __init__(self, adapter: PolicyAdapter) -> None:
        self.adapter = adapter
        self.seconds_last = 0.0
        self.stages_last: dict[str, float] = {}

    def __call__(self, packet: ObservationPacket, request_id: int, request_tick: int) -> np.ndarray:
        started = time.perf_counter()
        actions = self.adapter.infer(packet, request_id, request_tick)
        self.seconds_last = time.perf_counter() - started
        self.stages_last = dict(getattr(self.adapter, "last_stage_seconds", {}))
        return actions


def build_components(
    *,
    instruction: str,
    obs_delay_ticks: int,
    latency_schedule: LatencySchedule,
    policy_adapter: PolicyAdapter,
    action_spec: ActionSpec,
    switch_rule: SwitchRule,
    segment_ticks: int,
    trigger: Trigger,
    config_hash: str = "",
) -> EpisodeComponents:
    sensor_queue = SensorQueue(delay_ticks=obs_delay_ticks, instruction=instruction)
    scheduler = InferenceScheduler(
        schedule=latency_schedule,
        inference_fn=_TimedInference(policy_adapter),
        action_spec=action_spec,
        checkpoint=getattr(policy_adapter, "checkpoint_label", ""),
        config_hash=config_hash,
    )
    executor = PlanExecutor(action_spec, switch_rule, segment_ticks=segment_ticks)
    view = PolicyView(instruction=instruction)
    return EpisodeComponents(
        sensor_queue=sensor_queue,
        scheduler=scheduler,
        executor=executor,
        trigger=trigger,
        view=view,
    )


def run_episode(
    *,
    env: LatencyEnv,
    policy_adapter: PolicyAdapter,
    trigger: Trigger,
    action_spec: ActionSpec,
    obs_delay_ticks: int,
    latency_schedule: LatencySchedule,
    switch_rule: SwitchRule,
    segment_ticks: int = 1,
    max_ticks: int = 300,
    init_index: int,
    seed: int,
    config_hash: str = "",
    render_callback: Any = None,
) -> EpisodeTrace:
    """Run one episode under the discrete-event latency semantics."""

    components = build_components(
        instruction=env.instruction,
        obs_delay_ticks=obs_delay_ticks,
        latency_schedule=latency_schedule,
        policy_adapter=policy_adapter,
        action_spec=action_spec,
        switch_rule=switch_rule,
        segment_ticks=segment_ticks,
        trigger=trigger,
        config_hash=config_hash,
    )
    policy_adapter.reset()
    trigger.reset()
    raw_obs = env.start_episode(init_index=init_index, seed=seed)

    events: list[StepEvent] = []
    requests: list[RequestRecord] = []
    actions_executed: list[np.ndarray] = []
    success_flags: list[bool] = []
    first_success_step: int | None = None
    termination_reason = "timeout"
    episode_started = time.monotonic()
    tick = 0
    while tick < max_ticks:
        # (1) observation snapshot of the state at the start of this tick
        components.sensor_queue.capture(tick, raw_obs)
        # (2) release due observations into the policy view
        arrived = components.sensor_queue.release_due(tick)
        for packet in arrived:
            components.view.on_packet_arrived(packet)
        # (3) release due inference results as activatable plans
        completed_ids: list[int] = []
        for plan in components.scheduler.release_due(tick):
            components.executor.offer(plan)
            completed_ids.append(plan.plan_id)
        # (4)+(5) trigger decision; at most one request per tick
        request_id: int | None = None
        request_reason: str | None = None
        suppressed: str | None = None
        trigger_detail: dict[str, float] = {}
        if not components.scheduler.in_flight_request:
            decision = trigger.evaluate(tick, components.view)
            trigger_detail = dict(decision.detail)
            if decision.fire:
                packet = components.view.latest_packet
                assert packet is not None
                request = components.scheduler.submit(tick, packet, decision.reason)
                request_id = request.request_id
                request_reason = decision.reason
                trigger.on_request_submitted(tick, packet)
                requests.append(
                    RequestRecord(
                        request_id=request.request_id,
                        request_tick=tick,
                        input_capture_tick=request.input_capture_tick,
                        completion_tick=request.completion_tick,
                        reason=decision.reason,
                        inference_seconds=components.scheduler.inference_fn.seconds_last,
                        stage_seconds=dict(components.scheduler.inference_fn.stages_last),
                    )
                )
                # zero-delay requests may act within this same tick
                for plan in components.scheduler.release_due(tick):
                    components.executor.offer(plan)
                    completed_ids.append(plan.plan_id)
            else:
                suppressed = decision.reason
        # (6) select exactly one action
        action, source, fallback_reason, active_plan = components.executor.next_action(tick)
        components.view.on_command_sent(tick, action)
        # (7) one control step; evaluator-only info never feeds back this tick
        raw_obs, _reward, terminated, _truncated, info = env.step(action)
        is_success = bool(info.get("is_success", False))
        if is_success and first_success_step is None:
            first_success_step = tick
        success_flags.append(is_success)

        activated_plan_id = None
        if active_plan is not None and active_plan.activated_tick == tick:
            activated_plan_id = active_plan.plan_id
        events.append(
            StepEvent(
                tick=tick,
                arrived_packet_ids=[p.packet_id for p in arrived],
                latest_packet_id=(
                    components.view.latest_packet.packet_id
                    if components.view.latest_packet is not None
                    else None
                ),
                request_submitted=request_id is not None,
                request_id=request_id,
                request_reason=request_reason,
                trigger_suppressed_reason=suppressed,
                trigger_detail=trigger_detail,
                completed_request_ids=completed_ids,
                activated_plan_id=activated_plan_id,
                action_source=str(source.value),
                action_summary=[float(v) for v in action],
                fallback_reason=fallback_reason,
                active_plan_id=active_plan.plan_id if active_plan is not None else None,
                active_plan_age_ticks=(
                    active_plan.plan_age_ticks(tick) if active_plan is not None else None
                ),
                input_obs_age_ticks=(
                    components.view.latest_packet.capture_age_ticks(tick)
                    if components.view.latest_packet is not None
                    else None
                ),
                in_flight=components.scheduler.in_flight_request,
                wall_time_s=time.monotonic(),
            )
        )
        actions_executed.append(np.array(action, copy=True))
        if render_callback is not None:
            render_callback(tick)
        if source is ActionSource.FALLBACK_INVALID:
            termination_reason = "invalid_action"
            break
        if terminated:
            termination_reason = "success" if is_success else "env_done"
            break
        tick += 1

    success = termination_reason == "success" or any(success_flags)
    inference_total = float(sum(record.inference_seconds for record in requests))
    outcome = EpisodeOutcome(
        success=bool(success),
        termination_reason=termination_reason,
        steps=len(events),
        first_success_step=first_success_step,
        n_requests=len(requests),
        inference_seconds_total=inference_total,
        wall_seconds=time.monotonic() - episode_started,
    )
    return EpisodeTrace(
        outcome=outcome,
        events=events,
        requests=requests,
        actions=np.stack(actions_executed) if actions_executed else np.zeros((0, action_spec.dim)),
        success_flags=success_flags,
    )
