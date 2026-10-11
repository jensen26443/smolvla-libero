"""Action-plan consumption, switching, and exhaustion fallback."""

from __future__ import annotations

import numpy as np

from .contracts import (
    ActionPlan,
    ActionSource,
    ActionSpec,
    SwitchRule,
)


class PlanExecutor:
    """Executes one action per tick under a single, explicit switch rule.

    B1/M/R0 use ``IMMEDIATE``: a plan released at tick ``t`` becomes the active
    plan at tick ``t`` (the next executable boundary) and is consumed from its
    first action. B2 uses ``SEGMENT_BOUNDARY``: activation waits for the next
    multiple of ``segment_ticks``; while waiting, a newer released plan replaces
    an older still-pending one (last result wins), and the previous active plan
    keeps executing. Exhausted or absent plans emit the shared fallback action;
    delta semantics forbid re-emitting a previous delta, so the fallback is the
    zero-delta dummy action with an explicit legal gripper command.
    """

    def __init__(self, action_spec: ActionSpec, switch_rule: SwitchRule, segment_ticks: int = 1) -> None:
        if switch_rule is SwitchRule.SEGMENT_BOUNDARY and segment_ticks < 1:
            raise ValueError("segment_ticks must be >= 1")
        self.action_spec = action_spec
        self.switch_rule = switch_rule
        self.segment_ticks = int(segment_ticks)
        self.active: ActionPlan | None = None
        self.pending: list[ActionPlan] = []
        self.n_switches = 0
        self.n_exhausted_events = 0
        self.n_dropped_pending = 0

    def offer(self, plan: ActionPlan) -> None:
        """Register a plan whose result just became available this tick."""

        if self.switch_rule is SwitchRule.IMMEDIATE:
            self._activate(plan)
        else:
            if self.pending:
                # A newer result supersedes an older still-waiting one.
                self.n_dropped_pending += 1
            self.pending = [plan]

    def _activate(self, plan: ActionPlan) -> None:
        plan.activated_tick = None  # set by next_action with the actual tick
        self.active = plan
        self.n_switches += 1

    def next_action(self, tick: int) -> tuple[np.ndarray, ActionSource, str | None, ActionPlan | None]:
        """Select the action for ``tick``.

        Returns (action, source, fallback_reason, active_plan_after). The action
        is a fresh array owned by the caller. Non-finite plan actions fall back
        to the dummy action for that tick (episode-level accounting of invalid
        actions happens in the runner).
        """

        if (
            self.switch_rule is SwitchRule.SEGMENT_BOUNDARY
            and self.pending
            and tick % self.segment_ticks == 0
        ):
            self._activate(self.pending.pop(0))
        if self.active is not None and self.active.activated_tick is None:
            self.active.activated_tick = tick

        if self.active is None:
            return (
                np.array(self.action_spec.fallback, dtype=np.float64),
                ActionSource.FALLBACK_NO_PLAN,
                "no_plan_available",
                None,
            )
        if self.active.exhausted:
            self.n_exhausted_events += 1
            return (
                np.array(self.action_spec.fallback, dtype=np.float64),
                ActionSource.FALLBACK_EXHAUSTED,
                "plan_exhausted",
                self.active,
            )
        action = np.asarray(self.active.next_action(), dtype=np.float64)
        if not np.isfinite(action).all():
            return (
                np.array(self.action_spec.fallback, dtype=np.float64),
                ActionSource.FALLBACK_INVALID,
                "non_finite_action",
                self.active,
            )
        return action, ActionSource.PLAN, None, self.active

    def reset(self) -> None:
        self.active = None
        self.pending.clear()
        self.n_switches = 0
        self.n_exhausted_events = 0
        self.n_dropped_pending = 0
