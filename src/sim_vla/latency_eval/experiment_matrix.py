"""Experiment matrix expansion, batch execution, and resume.

Trials are (task, method, latency condition, initialization) tuples with a
deterministic order and stable ids. Each trial publishes its result
atomically (temporary directory renamed into place) so an interrupted batch
leaves only clean states behind. Resume re-runs anything not completed;
episodes restart from the beginning (environment and model internals are not
checkpointed mid-episode), attempts accumulate, and infrastructure failures
are retried under one explicit rule instead of being dropped silently.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..protocol import dump_json
from .config import ExperimentConfig, JitterConfig, MethodConfig, config_hash, write_frozen_spec
from .contracts import ActionSpec, SwitchRule
from .inference_scheduler import FixedDelay, JitteredDelay
from .runner import EpisodeTrace, run_episode
from .triggers import EveryStepTrigger, FeedbackTrigger, FixedPeriodTrigger


@dataclass(frozen=True)
class TrialSpec:
    trial_id: str
    index: int
    task_key: str
    method_name: str
    method_kind: str
    latency_name: str
    obs_delay_ticks: int
    inference_delay_ticks: int
    jitter: JitterConfig | None
    init_index: int
    seed: int
    record_video: bool


def expand_trials(config: ExperimentConfig) -> list[TrialSpec]:
    trials: list[TrialSpec] = []
    index = 0
    for task_key in config.tasks:
        for method in config.methods:
            if method.task_keys and task_key not in method.task_keys:
                continue
            for condition in config.latency_conditions:
                if method.latency_names and condition.name not in method.latency_names:
                    continue
                for rank, (init_index, seed) in enumerate(config.inits.pairs()):
                    trials.append(
                        TrialSpec(
                            trial_id=(
                                f"{task_key}__{method.name}__{condition.name}__i{init_index:02d}"
                            ),
                            index=index,
                            task_key=task_key,
                            method_name=method.name,
                            method_kind=method.kind,
                            latency_name=condition.name,
                            obs_delay_ticks=condition.obs_delay_ticks,
                            inference_delay_ticks=condition.inference_delay_ticks,
                            jitter=condition.jitter,
                            init_index=init_index,
                            seed=seed,
                            record_video=(
                                config.video.record and rank < config.video.trials_per_condition
                            ),
                        )
                    )
                    index += 1
    return trials


def build_trigger(method: MethodConfig):
    if method.kind == "r0_every_step":
        return EveryStepTrigger()
    if method.kind == "b1_fixed_period":
        return FixedPeriodTrigger(period_ticks=method.period_ticks or 1)
    if method.kind == "b2_fixed_period_conservative":
        return FixedPeriodTrigger(period_ticks=method.period_ticks or 1)
    if method.kind == "m_feedback":
        fb = method.feedback
        assert fb is not None
        return FeedbackTrigger(
            diff_threshold_high=fb.diff_threshold_high,
            diff_threshold_low=fb.diff_threshold_low,
            min_interval_ticks=fb.min_interval_ticks,
            max_interval_ticks=fb.max_interval_ticks,
            image_key=fb.image_key,
            downsample_factor=fb.downsample_factor,
        )
    raise ValueError(f"unknown method kind {method.kind}")


def switch_rule_of(method: MethodConfig) -> tuple[SwitchRule, int]:
    if method.kind == "b2_fixed_period_conservative":
        return SwitchRule.SEGMENT_BOUNDARY, int(method.segment_ticks or 1)
    return SwitchRule.IMMEDIATE, 1


def jitter_curve_path(experiment_dir: Path, condition_name: str, jitter: JitterConfig) -> Path:
    return experiment_dir / "latency_curves" / f"{condition_name}.npz"


def generate_jitter_curve(jitter: JitterConfig, length: int) -> np.ndarray:
    if jitter.distribution != "uniform":
        raise ValueError(f"unsupported jitter distribution {jitter.distribution}")
    rng = np.random.default_rng(jitter.seed)
    return rng.uniform(jitter.low, jitter.high, size=length)


def load_or_create_curve(path: Path, jitter: JitterConfig, length: int) -> np.ndarray:
    if path.is_file():
        with np.load(path) as data:
            curve = np.asarray(data["multipliers"])
        if len(curve) < length:
            raise ValueError(
                f"stored jitter curve at {path} has {len(curve)} ticks, need >= {length}"
            )
        return curve[:length]
    path.parent.mkdir(parents=True, exist_ok=True)
    curve = generate_jitter_curve(jitter, length)
    # numpy appends .npz to names that lack it, so the temp name must end in .npz
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, multipliers=curve, seed=jitter.seed, low=jitter.low, high=jitter.high)
    tmp.replace(path)
    return curve


def trace_summary_stats(trace: EpisodeTrace, tick_seconds: float) -> dict[str, Any]:
    events = trace.events
    obs_ages = [e.input_obs_age_ticks for e in events if e.input_obs_age_ticks is not None]
    plan_ages = [e.active_plan_age_ticks for e in events if e.active_plan_age_ticks is not None]
    reasons: dict[str, int] = {}
    for record in trace.requests:
        reasons[record.reason] = reasons.get(record.reason, 0) + 1
    actions = trace.actions
    if len(actions) > 1:
        deltas = np.linalg.norm(np.diff(actions, axis=0), axis=1)
        action_change_mean = float(deltas.mean())
        action_change_max = float(deltas.max())
    else:
        action_change_mean = action_change_max = None
    fallback_ticks = sum(1 for e in events if e.action_source.startswith("fallback"))
    sim_seconds = len(events) * tick_seconds
    return {
        "input_obs_age_ticks_mean": float(np.mean(obs_ages)) if obs_ages else None,
        "input_obs_age_ticks_max": int(np.max(obs_ages)) if obs_ages else None,
        "active_plan_age_ticks_mean": float(np.mean(plan_ages)) if plan_ages else None,
        "active_plan_age_ticks_max": int(np.max(plan_ages)) if plan_ages else None,
        "fallback_ticks": fallback_ticks,
        "fallback_fraction": fallback_ticks / len(events) if events else None,
        "trigger_reasons": reasons,
        "action_change_mean_per_tick": action_change_mean,
        "action_change_max_per_tick": action_change_max,
        "requests_per_sim_second": (trace.outcome.n_requests / sim_seconds) if sim_seconds else None,
        "requests_per_success": (
            trace.outcome.n_requests / trace.outcome.steps if trace.outcome.steps else None
        ),
    }


def _write_trial_artifacts(
    trial_dir: Path,
    *,
    result: dict[str, Any],
    trace: EpisodeTrace,
    tick_seconds: float,
) -> None:
    tmp_dir = trial_dir.parent / (trial_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    with (tmp_dir / "events.jsonl").open("w", encoding="utf-8") as handle:
        for event in trace.events:
            handle.write(json.dumps(event.to_json_dict(), ensure_ascii=False) + "\n")
        for record in trace.requests:
            handle.write(json.dumps({"request": record.to_json_dict()}, ensure_ascii=False) + "\n")
    np.savez_compressed(
        tmp_dir / "arrays.npz",
        actions=trace.actions,
        success_flags=np.asarray(trace.success_flags, dtype=bool),
    )
    result = dict(result)
    result.setdefault("summary_stats", trace_summary_stats(trace, tick_seconds))
    dump_json(result, tmp_dir / "result.json")
    if trial_dir.exists():
        shutil.rmtree(trial_dir)  # supersede a partial/older publication
    tmp_dir.replace(trial_dir)


def _hash_directory(path: Path, max_bytes: int = 64 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    total = 0
    for file in sorted(path.rglob("*")):
        if file.is_file():
            total += file.stat().st_size
            if total > max_bytes:
                break
            digest.update(file.name.encode())
            digest.update(file.read_bytes())
    return digest.hexdigest()[:16]


class ExperimentRunner:
    """Executes trials serially with resume; owns env/policy lifecycles."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        output_root: str | Path = "runs",
        env_factory=None,
        policy_factory=None,
        resume: bool = True,
        tick_seconds: float = 0.05,
    ) -> None:
        self.config = config
        self.experiment_dir = Path(output_root) / config.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.trials = expand_trials(config)
        self.resume = resume
        self.tick_seconds = tick_seconds
        self._env_factory = env_factory
        self._policy_factory = policy_factory
        self._env = None
        self._env_task = None
        self._env_trial_count = 0
        self._policy = None
        self.checkpoint_hash = _hash_directory(Path(config.checkpoint))
        self.code_hash = _code_fingerprint()
        write_frozen_spec(
            config,
            self.experiment_dir / "experiment_spec.yaml",
            extra={
                "checkpoint_hash": self.checkpoint_hash,
                "code_hash": self.code_hash,
                "n_trials": len(self.trials),
                "created_unix": time.time(),
            },
        )

    # -- lifecycle ------------------------------------------------------------

    def policy_adapter(self):
        if self._policy is None:
            if self._policy_factory is not None:
                self._policy = self._policy_factory()
            else:
                from .policy_adapter import SmolVLAPolicyAdapter

                self._policy = SmolVLAPolicyAdapter(
                    checkpoint=self.config.checkpoint,
                    device=self.config.device,
                    task_ids_for_cfg=tuple(
                        # distinct task ids for the policy env-config features
                        dict.fromkeys(self._suite_task_ids())
                    ),
                )
                self._policy.warmup()
        return self._policy

    def _suite_task_ids(self) -> list[int]:
        from ..tasks import TASK_BY_KEY

        return [TASK_BY_KEY[key].suite_task_id for key in self.config.tasks]

    def env_for(self, task_key: str):
        if self._env is not None and (self._env_task != task_key or self._env_trial_count >= self.config.env_rebuild_every):
            self._env.close()
            self._env = None
            self._env_trial_count = 0
        if self._env is None:
            if self._env_factory is not None:
                self._env = self._env_factory(task_key)
            else:
                from .env_adapter import LiberoLatencyEnv

                self._env = LiberoLatencyEnv(task_key=task_key, episode_length=self.config.max_ticks)
            self._env_task = task_key
        return self._env

    def close(self) -> None:
        if self._env is not None:
            closer = getattr(self._env, "close", None)
            if callable(closer):
                closer()
            self._env = None

    # -- state ----------------------------------------------------------------

    def trial_dir(self, trial: TrialSpec) -> Path:
        return self.experiment_dir / "trials" / trial.trial_id

    def load_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        for trial in self.trials:
            result_path = self.trial_dir(trial) / "result.json"
            if result_path.is_file():
                try:
                    payload = json.loads(result_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    state[trial.trial_id] = {"status": "pending"}
                    continue
                state[trial.trial_id] = {"status": "completed", "result": payload}
            else:
                state[trial.trial_id] = {"status": "pending"}
        # Infrastructure failures live in failures.jsonl (no result.json is
        # published for them); restore them so the retry budget survives
        # process restarts.
        failed_log = self.experiment_dir / "failures.jsonl"
        if failed_log.is_file():
            for line in failed_log.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                trial_id = entry.get("trial_id")
                if trial_id in state and state[trial_id].get("status") != "completed":
                    state[trial_id] = {
                        "status": "failed_infrastructure",
                        "error": entry.get("error"),
                        "attempt": int(entry.get("attempt", 1)),
                    }
        return state

    # -- execution ------------------------------------------------------------

    def _schedule_for(self, trial: TrialSpec):
        if trial.jitter is None:
            return FixedDelay(trial.inference_delay_ticks)
        curve = load_or_create_curve(
            jitter_curve_path(self.experiment_dir, trial.latency_name, trial.jitter),
            trial.jitter,
            length=max(self.config.max_ticks + 8, trial.jitter.length_ticks),
        )
        return JitteredDelay(base_ticks=trial.inference_delay_ticks, multipliers=curve)

    def run_trial(self, trial: TrialSpec, attempt: int) -> dict[str, Any]:
        method = next(m for m in self.config.methods if m.name == trial.method_name)
        env = self.env_for(trial.task_key)
        policy = self.policy_adapter()
        switch_rule, segment = switch_rule_of(method)
        recorder = None
        render_callback = None
        if trial.record_video:
            from .env_adapter import VideoRecorder

            recorder = VideoRecorder(env, fps=self.config.video.fps)
            recorder.reset()

            def render_callback(tick: int) -> None:
                recorder.capture(tick)

        policy.set_episode_seed(trial.seed)
        trace = run_episode(
            env=env,
            policy_adapter=policy,
            trigger=build_trigger(method),
            action_spec=ActionSpec(horizon=policy.chunk_size if hasattr(policy, "chunk_size") else 50),
            obs_delay_ticks=trial.obs_delay_ticks,
            latency_schedule=self._schedule_for(trial),
            switch_rule=switch_rule,
            segment_ticks=segment,
            max_ticks=self.config.max_ticks,
            init_index=trial.init_index,
            seed=trial.seed,
            config_hash=config_hash(self.config),
            render_callback=render_callback,
        )
        video_path = None
        if recorder is not None and recorder.frames:
            video_path = recorder.save(self.trial_dir(trial).parent / (trial.trial_id + ".mp4"))
        result = {
            "trial_id": trial.trial_id,
            "experiment_id": self.config.experiment_id,
            "config_hash": config_hash(self.config),
            "code_hash": self.code_hash,
            "checkpoint": self.config.checkpoint,
            "checkpoint_hash": self.checkpoint_hash,
            "task": trial.task_key,
            "method": trial.method_name,
            "method_kind": trial.method_kind,
            "latency": trial.latency_name,
            "obs_delay_ticks": trial.obs_delay_ticks,
            "inference_delay_ticks": trial.inference_delay_ticks,
            "jitter": None if trial.jitter is None else str(jitter_curve_path(self.experiment_dir, trial.latency_name, trial.jitter)),
            "init_index": trial.init_index,
            "seed": trial.seed,
            "attempt": attempt,
            "status": "completed",
            "success": bool(trace.outcome.success),
            "termination_reason": trace.outcome.termination_reason,
            "failure_type": None if trace.outcome.success else trace.outcome.termination_reason,
            "steps": trace.outcome.steps,
            "first_success_step": trace.outcome.first_success_step,
            "n_requests": trace.outcome.n_requests,
            "inference_seconds_total": trace.outcome.inference_seconds_total,
            "wall_seconds": trace.outcome.wall_seconds,
            "events_path": "events.jsonl",
            "arrays_path": "arrays.npz",
            "video_path": str(video_path.name) if video_path else None,
            "tick_seconds": self.tick_seconds,
        }
        _write_trial_artifacts(
            self.trial_dir(trial),
            result=result,
            trace=trace,
            tick_seconds=self.tick_seconds,
        )
        if recorder is not None:
            recorder.reset()
        self._env_trial_count += 1
        return result

    def run(self, *, limit: int | None = None, only_pending: bool = True) -> dict[str, Any]:
        state = self.load_state()
        executed = 0
        attempts_by_trial = self._read_attempts(state)
        started = time.time()
        try:
            for trial in self.trials:
                entry = state.get(trial.trial_id, {"status": "pending"})
                if entry.get("status") == "completed" and only_pending:
                    continue
                if entry.get("status") == "failed_infrastructure":
                    prior = attempts_by_trial.get(trial.trial_id, 0)
                    if prior >= self.config.max_attempts:
                        continue
                attempt = attempts_by_trial.get(trial.trial_id, 0) + 1
                attempts_by_trial[trial.trial_id] = attempt
                try:
                    result = self.run_trial(trial, attempt)
                    state[trial.trial_id] = {"status": "completed", "result": result}
                except Exception as exc:  # noqa: BLE001 - infrastructure failure path
                    state[trial.trial_id] = {
                        "status": "failed_infrastructure",
                        "error": f"{type(exc).__name__}: {exc}",
                        "attempt": attempt,
                    }
                    # Environment state is now unknown: force a rebuild.
                    if self._env is not None:
                        try:
                            self._env.close()
                        except Exception:  # noqa: BLE001, S110 - best-effort cleanup
                            pass
                        self._env = None
                        self._env_trial_count = 0
                executed += 1
                self._write_state_snapshot(state, executed)
                if limit is not None and executed >= limit:
                    break
        finally:
            self.close()
        completed = sum(1 for v in state.values() if v.get("status") == "completed")
        failed = sum(1 for v in state.values() if v.get("status") == "failed_infrastructure")
        return {
            "experiment_id": self.config.experiment_id,
            "trials_total": len(self.trials),
            "completed": completed,
            "failed_infrastructure": failed,
            "executed_this_run": executed,
            "elapsed_seconds": time.time() - started,
            "state": {k: v.get("status") for k, v in state.items()},
        }

    def _read_attempts(self, state: dict[str, Any]) -> dict[str, int]:
        attempts: dict[str, int] = {}
        failed_log = self.experiment_dir / "failures.jsonl"
        if failed_log.is_file():
            for line in failed_log.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    attempts[entry["trial_id"]] = max(
                        attempts.get(entry["trial_id"], 0), int(entry["attempt"])
                    )
        for trial_id, entry in state.items():
            if entry.get("status") == "failed_infrastructure":
                attempts[trial_id] = max(attempts.get(trial_id, 0), int(entry.get("attempt", 0)))
        return attempts

    def _write_state_snapshot(self, state: dict[str, Any], executed: int) -> None:
        snapshot = {
            "experiment_id": self.config.experiment_id,
            "updated_unix": time.time(),
            "executed_this_run": executed,
            "statuses": {k: v.get("status") for k, v in state.items()},
        }
        tmp = self.experiment_dir / "matrix.json.tmp"
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, ensure_ascii=False, indent=1)
        tmp.replace(self.experiment_dir / "matrix.json")
        failures = {
            k: {"trial_id": k, **{kk: vv for kk, vv in v.items() if kk != "result"}}
            for k, v in state.items()
            if v.get("status") == "failed_infrastructure"
        }
        if failures:
            # Rewrite (not append): every snapshot reflects the full current
            # failure set, deduplicated by trial with its latest attempt.
            tmp_failures = self.experiment_dir / "failures.jsonl.tmp"
            with tmp_failures.open("w", encoding="utf-8") as handle:
                for entry in sorted(failures.values(), key=lambda e: e["trial_id"]):
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            tmp_failures.replace(self.experiment_dir / "failures.jsonl")


def _code_fingerprint() -> str:
    package_root = Path(__file__).parent
    digest = hashlib.sha256()
    for file in sorted(package_root.glob("*.py")):
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()[:16]


def run_experiment(
    config: ExperimentConfig,
    *,
    output_root: str | Path = "runs",
    resume: bool = True,
    limit: int | None = None,
    env_factory=None,
    policy_factory=None,
) -> dict[str, Any]:
    runner = ExperimentRunner(
        config,
        output_root=output_root,
        env_factory=env_factory,
        policy_factory=policy_factory,
        resume=resume,
    )
    try:
        return runner.run(limit=limit, only_pending=resume)
    finally:
        runner.close()
