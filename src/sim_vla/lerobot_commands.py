"""Build explicit LeRobot 0.5.1 commands without executing them implicitly."""

from __future__ import annotations

import shlex
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .assets import model_snapshot_dir, project_root
from .protocol import DATASET_REPO, DATASET_REVISION, LEROBOT_VERSION, SEED
from .tasks import selected_suite_ids


@dataclass(frozen=True)
class CommandSpec:
    executable: str
    args: tuple[str, ...]
    expected_version: str = LEROBOT_VERSION

    @property
    def argv(self) -> list[str]:
        return [self.executable, *self.args]

    def shell(self) -> str:
        return shlex.join(self.argv)


def _episodes_arg(episodes: Iterable[int] | None) -> str | None:
    if episodes is None:
        return None
    values = sorted({int(value) for value in episodes})
    if any(value < 0 for value in values):
        raise ValueError("episode IDs must be non-negative")
    return "[" + ",".join(str(value) for value in values) + "]"


def build_train_command(
    *,
    output_dir: str | Path,
    episodes: Iterable[int] | None = None,
    task_ids: Iterable[int] | None = None,
    steps: int = 10_000,
    eval_freq: int = 2_000,
    save_freq: int = 2_000,
    seed: int = SEED,
    policy_path: str | Path | None = None,
    dataset_root: str | Path | None = None,
    executable: str = "lerobot-train",
) -> CommandSpec:
    if steps <= 0 or eval_freq <= 0 or save_freq <= 0:
        raise ValueError("steps and checkpoint/evaluation frequencies must be positive")
    task_ids = list(selected_suite_ids()) if task_ids is None else sorted({int(value) for value in task_ids})
    # LeRobot 0.5.1 has no valid --policy.revision CLI field.  Use the local
    # snapshot materialized by `sim-vla prepare-assets --kind model`, which is
    # how the protocol's model commit remains pinned at the official entrypoint.
    pinned_policy_path = Path(policy_path) if policy_path is not None else model_snapshot_dir()
    pinned_dataset_root = (
        Path(dataset_root) if dataset_root is not None else project_root() / ".cache" / "lerobot" / "libero"
    )
    args = [
        f"--policy.path={pinned_policy_path}",
        f"--dataset.repo_id={DATASET_REPO}",
        f"--dataset.root={pinned_dataset_root}",
        f"--dataset.revision={DATASET_REVISION}",
        "--dataset.use_imagenet_stats=false",
        "--env.type=libero",
        "--env.task=libero_goal",
        f"--env.task_ids={task_ids}",
        "--env.control_mode=relative",
        f"--output_dir={Path(output_dir)}",
        f"--job_name=libero_goal_smolvla_seed{seed}",
        f"--seed={seed}",
        "--policy.device=cuda",
        "--policy.use_amp=false",
        "--policy.push_to_hub=false",
        "--batch_size=1",
        "--num_workers=0",
        f"--steps={steps}",
        f"--eval_freq={eval_freq}",
        f"--save_freq={save_freq}",
        "--eval.n_episodes=10",
        "--eval.batch_size=1",
        "--eval.use_async_envs=false",
        "--wandb.enable=false",
    ]
    episode_arg = _episodes_arg(episodes)
    if episode_arg is not None:
        args.insert(4, f"--dataset.episodes={episode_arg}")
    return CommandSpec(executable, tuple(args))


def build_eval_command(
    *,
    checkpoint: str | Path,
    output_dir: str | Path,
    task_ids: Iterable[int] | None = None,
    episodes: int = 20,
    seed: int = SEED,
    policy_path: str | Path | None = None,
    executable: str = "lerobot-eval",
) -> CommandSpec:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    task_ids = list(selected_suite_ids()) if task_ids is None else sorted({int(value) for value in task_ids})
    selected_checkpoint = Path(policy_path) if policy_path is not None else checkpoint
    args = (
        f"--policy.path={selected_checkpoint}",
        "--policy.device=cuda",
        "--policy.use_amp=false",
        "--env.type=libero",
        "--env.task=libero_goal",
        f"--env.task_ids={task_ids}",
        "--env.control_mode=relative",
        f"--eval.n_episodes={episodes}",
        "--eval.batch_size=1",
        "--env.max_parallel_tasks=1",
        f"--seed={seed}",
        f"--output_dir={Path(output_dir)}",
    )
    return CommandSpec(executable, args)
