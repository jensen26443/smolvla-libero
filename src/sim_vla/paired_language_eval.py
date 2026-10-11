"""Quantitative E3/E4 evaluation on a shared LIBERO initialization set.

LIBERO Goal stores a different ``*.pruned_init`` file for each task even
though the selected bowl tasks share the same tabletop geometry.  This module
keeps the official BDDL task and success predicate for each target while
injecting one frozen initialization array into every task environment.  The
result is a paired, target-specific evaluator instead of a cross-task score
comparison with mismatched initial states.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .assets import libero_assets_dir, project_root
from .language_alignment import _array_digest, _nested_array_digests
from .protocol import PROTOCOL_VERSION, SEED, dump_json
from .runtime import import_libero
from .tasks import TASK_BY_KEY, TaskSpec, selected_tasks

ANCHOR_TASK_KEY = "goal_bowl_plate"
TARGET_TASK_KEYS = ("goal_bowl_plate", "goal_bowl_stove", "goal_bowl_cabinet_top")
TARGET_TASK_IDS = tuple(TASK_BY_KEY[key].suite_task_id for key in TARGET_TASK_KEYS)


def _load_init_states(path: Path) -> np.ndarray:
    import torch

    states = np.asarray(torch.load(path, weights_only=False))  # nosec B614 - pinned local asset
    if states.ndim != 2 or states.shape[0] < 30:
        raise ValueError(f"shared initialization file must contain >=30 states, got {states.shape} at {path}")
    return states


def _path_for_task(task: Any, get_libero_path: Any, kind: str) -> Path:
    if kind == "bddl":
        return Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    if kind == "init_states":
        return Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    raise ValueError(f"unknown LIBERO path kind: {kind}")


def make_shared_env_class(base_cls: type) -> type:
    """Create a LiberoEnv subclass with shared-init and episode logging."""

    class _SharedInitLiberoEnv(base_cls):
        def __init__(self, *args: Any, shared_init_states: np.ndarray, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._shared_init_states = np.asarray(shared_init_states).copy()
            self._init_states = self._shared_init_states
            self._episode_active = False
            self._episode_step = 0
            self._current_init_index: int | None = None
            self._current_seed: int | None = None
            self._first_success_step: int | None = None
            self.episode_records: list[dict[str, Any]] = []

        def _finish_episode(self, *, terminal_success: bool | None = None) -> None:
            if not self._episode_active:
                return
            success = self._first_success_step is not None
            if terminal_success is not None:
                success = bool(terminal_success)
            self.episode_records.append(
                {
                    "task_id": int(self.task_id),
                    "task": str(self.task),
                    "instruction": str(self.task_description),
                    "seed": self._current_seed,
                    "initialization_index": self._current_init_index,
                    "steps": int(self._episode_step),
                    "success": bool(success),
                    "first_success_step": self._first_success_step,
                }
            )
            self._episode_active = False

        def reset(self, seed: int | None = None, **kwargs: Any):  # type: ignore[no-untyped-def]
            self._finish_episode()
            init_index = int(self.init_state_id % len(self._shared_init_states))
            observation, info = super().reset(seed=seed, **kwargs)
            self._episode_active = True
            self._episode_step = 0
            self._current_init_index = init_index
            self._current_seed = int(seed) if seed is not None else None
            self._first_success_step = None
            return observation, info

        def step(self, action: np.ndarray):  # type: ignore[no-untyped-def]
            if action.ndim != 1:
                raise ValueError(f"Expected action shape (action_dim,), got {action.shape}")
            raw_obs, reward, done, info = self._env.step(action)
            is_success = bool(self._env.check_success())
            self._episode_step += 1
            if is_success and self._first_success_step is None:
                self._first_success_step = self._episode_step
            terminated = bool(done or is_success)
            info.update(
                {
                    "task": self.task,
                    "task_id": self.task_id,
                    "done": bool(done),
                    "is_success": is_success,
                }
            )
            observation = self._format_raw_obs(raw_obs)
            if terminated:
                info["final_info"] = {
                    "task": self.task,
                    "task_id": self.task_id,
                    "done": bool(done),
                    "is_success": is_success,
                }
                self._finish_episode(terminal_success=is_success)
            return observation, reward, terminated, False, info

        def finalize_open_episode(self) -> None:
            self._finish_episode()

    _SharedInitLiberoEnv.__name__ = "SharedInitLiberoEnv"
    return _SharedInitLiberoEnv


def _make_vector_envs(
    *,
    suite: Any,
    task_specs: Sequence[TaskSpec],
    shared_states: np.ndarray,
    instructions: Mapping[int, str],
    episode_index: int,
    episode_length: int,
) -> dict[str, dict[int, Any]]:
    import gymnasium as gym
    from lerobot.envs.libero import LiberoEnv as BaseLiberoEnv

    env_class = make_shared_env_class(BaseLiberoEnv)
    result: dict[str, dict[int, gym.vector.VectorEnv]] = {"libero_goal": {}}
    for task_spec in task_specs:
        task_id = task_spec.suite_task_id

        def make_one(
            task_id: int = task_id,
            task_spec: TaskSpec = task_spec,
        ) -> Any:
            env = env_class(
                task_suite=suite,
                task_id=task_id,
                task_suite_name=task_spec.suite,
                episode_length=episode_length,
                camera_name="agentview_image,robot0_eye_in_hand_image",
                obs_type="pixels_agent_pos",
                observation_width=256,
                observation_height=256,
                init_states=True,
                episode_index=episode_index,
                n_envs=1,
                control_mode="relative",
                shared_init_states=shared_states,
            )
            env.task_description = instructions[task_id]
            return env

        result["libero_goal"][task_id] = gym.vector.SyncVectorEnv(
            # LiberoEnv's base implementation already provides final_info and
            # the evaluator calls reset between episodes. DISABLED prevents
            # Gymnasium from resetting a second time and skipping an init row.
            [make_one],
            autoreset_mode=gym.vector.AutoresetMode.DISABLED,
        )
    return result


def _close_envs(envs: Mapping[str, Mapping[int, Any]]) -> None:
    for group in envs.values():
        for env in group.values():
            env.close()


def _probe_shared_reset(
    *,
    suite: Any,
    task_specs: Sequence[TaskSpec],
    shared_states: np.ndarray,
    seed: int,
    episode_index: int,
) -> dict[str, Any]:
    """Verify exact observations after restoring the shared state once."""

    instructions = {task.suite_task_id: task.instruction.lower() for task in task_specs}
    envs = _make_vector_envs(
        suite=suite,
        task_specs=task_specs,
        shared_states=shared_states,
        instructions=instructions,
        episode_index=episode_index,
        episode_length=300,
    )
    records: dict[str, Any] = {}
    try:
        for task_spec in task_specs:
            vec = envs["libero_goal"][task_spec.suite_task_id]
            observation, info = vec.reset(seed=[seed])
            raw_env = vec.envs[0]
            records[task_spec.key] = {
                "task_id": task_spec.suite_task_id,
                "observation": _nested_array_digests(observation),
                "sim_state": _array_digest(raw_env._env.get_sim_state()),
                "info": {str(key): str(value) for key, value in info.items()},
            }
    finally:
        _close_envs(envs)
    first = records[task_specs[0].key]
    exact = all(
        record["observation"] == first["observation"] and record["sim_state"] == first["sim_state"]
        for record in records.values()
    )
    return {
        "status": "passed" if exact else "failed",
        "seed": seed,
        "episode_index": episode_index,
        "exact_observation_and_sim_state_match": exact,
        "records": records,
    }


def _prepare_policy(checkpoint: Path, env_cfg: Any, device: str):
    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.pretrained_path = checkpoint
    cfg.device = device
    if hasattr(cfg, "use_amp"):
        cfg.use_amp = False
    policy = make_policy(cfg=cfg, env_cfg=env_cfg)
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": {}},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=checkpoint,
        preprocessor_overrides=preprocessor_overrides,
    )
    return cfg, policy, preprocessor, postprocessor


def _run_rollout_condition(
    *,
    checkpoint: Path,
    suite: Any,
    task_specs: Sequence[TaskSpec],
    shared_states: np.ndarray,
    instructions: Mapping[int, str],
    output_dir: Path,
    condition: str,
    phase: str,
    start_seed: int,
    start_episode_index: int,
    episodes: int,
    device: str,
) -> dict[str, Any]:
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.scripts.lerobot_eval import eval_policy_all

    output_dir.mkdir(parents=True, exist_ok=True)
    env_cfg = LiberoEnvConfig(
        task="libero_goal",
        task_ids=[task.suite_task_id for task in task_specs],
        control_mode="relative",
        episode_length=300,
        max_parallel_tasks=1,
        observation_width=256,
        observation_height=256,
    )
    cfg, policy, preprocessor, postprocessor = _prepare_policy(checkpoint, env_cfg, device)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
    envs = _make_vector_envs(
        suite=suite,
        task_specs=task_specs,
        shared_states=shared_states,
        instructions=instructions,
        episode_index=start_episode_index,
        episode_length=300,
    )
    try:
        import torch

        with torch.no_grad():
            info = eval_policy_all(
                envs=envs,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=episodes,
                # Keep a short development video for visual inspection, while
                # avoiding hundreds of large MP4 files in the formal batch.
                max_episodes_rendered=min(10, episodes) if phase == "development" else 0,
                videos_dir=output_dir / "videos",
                start_seed=start_seed,
                max_parallel_tasks=1,
            )
        episode_records: list[dict[str, Any]] = []
        for task_spec in task_specs:
            vec = envs["libero_goal"][task_spec.suite_task_id]
            raw_env = vec.envs[0]
            raw_env.finalize_open_episode()
            for record in raw_env.episode_records:
                record = dict(record)
                record["task_key"] = task_spec.key
                record["phase"] = phase
                record["condition"] = condition
                episode_records.append(record)
    finally:
        _close_envs(envs)

    # ``eval_policy_all`` returns the stable official schema.  The additional
    # records retain shared initialization IDs and first-success timing.
    with (output_dir / "eval_info.json").open("w", encoding="utf-8") as handle:
        json.dump(info, handle, ensure_ascii=False, indent=2)
    dump_json(
        {
            "condition": condition,
            "phase": phase,
            "protocol_version": PROTOCOL_VERSION,
            "checkpoint": str(checkpoint),
            "task_keys": [task.key for task in task_specs],
            "task_ids": [task.suite_task_id for task in task_specs],
            "instructions": {str(task_id): instructions[task_id] for task_id in sorted(instructions)},
            "shared_init_anchor": ANCHOR_TASK_KEY,
            "shared_init_state_count": int(shared_states.shape[0]),
            "shared_init_state_shape": list(shared_states.shape),
            "episodes_per_task": episodes,
            "start_seed": start_seed,
            "seeds": list(range(start_seed, start_seed + episodes)),
            "start_initialization_index": start_episode_index,
            "initialization_indices": list(range(start_episode_index, start_episode_index + episodes)),
            "control_mode": "relative",
            "episode_length": 300,
            "success_predicate": "official native LIBERO Goal BDDL predicate for each target task",
            "customization": "only the shared initialization array is injected; BDDL geometry and target predicate remain native",
        },
        output_dir / "protocol.json",
    )
    dump_json({"records": episode_records, "count": len(episode_records)}, output_dir / "episode_records.json")
    return info


def _summary_from_eval(info: Mapping[str, Any], task_key_by_id: Mapping[int, str], condition: str) -> dict[str, Any]:
    from .statistics import summarize_trials

    records: list[dict[str, Any]] = []
    for task_result in info.get("per_task", []):
        task_id = int(task_result["task_id"])
        metrics = task_result["metrics"]
        for index, success in enumerate(metrics["successes"]):
            records.append(
                {
                    "condition": condition,
                    "task_key": task_key_by_id[task_id],
                    "episode_index": index,
                    "success": bool(success),
                    "sum_reward": float(metrics["sum_rewards"][index]),
                    "max_reward": float(metrics["max_rewards"][index]),
                    "valid_trial": True,
                    "slice": "development" if condition.endswith("_dev") else "final",
                }
            )
    summary = summarize_trials(records)
    summary["records"] = len(records)
    summary["source"] = "eval_info.json"
    return summary


def _condition_label(condition: str, phase: str) -> str:
    """Use explicit phase names in summaries instead of ambiguous abbreviations."""

    return f"{condition}_{'dev' if phase == 'development' else 'formal'}"


def run_paired_language_eval(
    *,
    checkpoint: str | Path,
    output_root: str | Path = "results",
    phase: str = "formal",
    episodes: int | None = None,
    seed: int = SEED,
    device: str = "cuda",
    assets: str | Path | None = None,
    overwrite: bool = False,
    conditions: str = "all",
    start_seed: int | None = None,
    start_initialization_index: int | None = None,
) -> dict[str, Any]:
    """Run E3 and/or E4 with a shared, frozen initialization array."""

    if phase not in {"development", "formal", "all"}:
        raise ValueError("phase must be development, formal, or all")
    if conditions not in {"all", "e3", "e4"}:
        raise ValueError("conditions must be all, e3, or e4")
    if phase == "development" and (start_seed is not None or start_initialization_index is not None):
        raise ValueError("start_seed/start_initialization_index overrides are only valid for formal phase")
    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    selected = selected_tasks(TARGET_TASK_KEYS)
    asset_path = Path(assets) if assets is not None else libero_assets_dir()
    if not asset_path.is_dir():
        raise FileNotFoundError(f"LIBERO assets not found at {asset_path}")
    import_libero(root=project_root(), assets=asset_path)
    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    anchor_task = suite.get_task(TASK_BY_KEY[ANCHOR_TASK_KEY].suite_task_id)
    anchor_init_path = _path_for_task(anchor_task, get_libero_path, "init_states")
    shared_states = _load_init_states(anchor_init_path)
    bddl_paths = {
        task.key: str(_path_for_task(suite.get_task(task.suite_task_id), get_libero_path, "bddl"))
        for task in selected
    }
    instructions = {
        task.suite_task_id: suite.get_task(task.suite_task_id).language.lower() for task in selected
    }
    empty_instructions = {TASK_BY_KEY[ANCHOR_TASK_KEY].suite_task_id: ""}
    phases = ("development", "formal") if phase == "all" else (phase,)
    results: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "audit_version": "sim-vla-paired-language-v1",
        "checkpoint": str(checkpoint_path),
        "anchor_task": ANCHOR_TASK_KEY,
        "anchor_init_states": str(anchor_init_path),
        "anchor_init_states_shape": list(shared_states.shape),
        "seed": seed,
        "bddl_paths": bddl_paths,
        "conditions": {},
        "created_unix": time.time(),
    }
    output_base = Path(output_root)
    for current_phase in phases:
        if current_phase == "development":
            count = 1 if episodes is None else episodes
            phase_start_seed = seed
            phase_start_index = 0
            e3_dir = output_base / "e3_shared_dev"
            e4_dir = output_base / "e4_empty_dev"
        else:
            count = 20 if episodes is None else episodes
            phase_start_seed = seed + 10 if start_seed is None else int(start_seed)
            phase_start_index = 10 if start_initialization_index is None else int(start_initialization_index)
            e3_dir = output_base / "e3_shared_formal"
            e4_dir = output_base / "e4_empty_formal"
        probe = _probe_shared_reset(
            suite=suite,
            task_specs=selected,
            shared_states=shared_states,
            seed=phase_start_seed,
            episode_index=phase_start_index,
        )
        if probe["status"] != "passed":
            raise RuntimeError(f"shared reset alignment failed for {current_phase}: {probe}")
        results.setdefault("shared_reset_probes", {})[current_phase] = probe
        if not overwrite:
            existing = [e3_dir / "eval_info.json", e4_dir / "eval_info.json"]
            if all(path.is_file() for path in existing):
                results["conditions"][current_phase] = {"status": "already_complete", "output": [str(e3_dir), str(e4_dir)]}
                continue
        phase_result: dict[str, Any] = {"status": "completed"}
        if conditions in {"all", "e3"}:
            e3_info = _run_rollout_condition(
                checkpoint=checkpoint_path,
                suite=suite,
                task_specs=selected,
                shared_states=shared_states,
                instructions=instructions,
                output_dir=e3_dir,
                condition="E3",
                phase=current_phase,
                start_seed=phase_start_seed,
                start_episode_index=phase_start_index,
                episodes=count,
                device=device,
            )
            e3_summary = _summary_from_eval(
                e3_info,
                {task.suite_task_id: task.key for task in selected},
                _condition_label("E3", current_phase),
            )
            dump_json(e3_summary, e3_dir / "summary.json")
            phase_result.update({"e3_output": str(e3_dir), "e3_summary": e3_summary["overall"]})
        e4_spec = (TASK_BY_KEY[ANCHOR_TASK_KEY],)
        if conditions in {"all", "e4"}:
            e4_info = _run_rollout_condition(
                checkpoint=checkpoint_path,
                suite=suite,
                task_specs=e4_spec,
                shared_states=shared_states,
                instructions=empty_instructions,
                output_dir=e4_dir,
                condition="E4_empty",
                phase=current_phase,
                start_seed=phase_start_seed,
                start_episode_index=phase_start_index,
                episodes=count,
                device=device,
            )
            e4_summary = _summary_from_eval(
                e4_info,
                {TASK_BY_KEY[ANCHOR_TASK_KEY].suite_task_id: ANCHOR_TASK_KEY},
                _condition_label("E4_empty", current_phase),
            )
            dump_json(e4_summary, e4_dir / "summary.json")
            phase_result.update({"e4_output": str(e4_dir), "e4_summary": e4_summary["overall"]})
        results["conditions"][current_phase] = phase_result
        dump_json(results, output_base / "paired_language_status.json")
    results["status"] = "completed"
    results["elapsed_seconds"] = time.time() - results["created_unix"]
    dump_json(results, output_base / "paired_language_status.json")
    return results


def paired_language_eval_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run quantitative E3/E4 LIBERO paired-language evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--phase", choices=("development", "formal", "all"), default="formal")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--assets")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--conditions", choices=("all", "e3", "e4"), default="all")
    parser.add_argument("--start-seed", type=int)
    parser.add_argument("--start-initialization-index", type=int)
    args = parser.parse_args(argv)
    result = run_paired_language_eval(
        checkpoint=args.checkpoint,
        output_root=args.output_root,
        phase=args.phase,
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
        assets=args.assets,
        overwrite=args.overwrite,
        conditions=args.conditions,
        start_seed=args.start_seed,
        start_initialization_index=args.start_initialization_index,
    )
    print(f"paired language evaluation {result['status']}; conditions={sorted(result['conditions'])}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(paired_language_eval_main())
