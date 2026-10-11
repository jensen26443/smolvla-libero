"""Frozen experiment protocol and common validation helpers.

The protocol is deliberately represented with standard-library dataclasses and
JSON-compatible values. This keeps manifests inspectable without importing
LeRobot, MuJoCo, or a model checkpoint.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "sim-vla-libero-v1"
LEROBOT_VERSION = "0.5.1"
MODEL_REPO = "HuggingFaceVLA/smolvla_libero"
MODEL_REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
DATASET_REPO = "HuggingFaceVLA/libero"
DATASET_REVISION = "86958911c0f959db2bbbdb107eb3e17c5f9c798e"
LIBERO_ASSETS_REPO = "lerobot/libero-assets"
LIBERO_ASSETS_REVISION = "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
SEED = 42


@dataclass(frozen=True)
class TrainingBudget:
    """One nested training-data condition."""

    name: str
    fraction: float
    steps: int = 10_000

    def __post_init__(self) -> None:
        if not self.name or not 0 < self.fraction <= 1:
            raise ValueError("training budget needs a name and fraction in (0, 1]")
        if self.steps <= 0:
            raise ValueError("training steps must be positive")


@dataclass(frozen=True)
class EvalProtocol:
    """Fixed evaluation settings shared by E0--E5."""

    dev_episodes_per_task: int = 10
    final_episodes_per_task: int = 20
    eval_batch_size: int = 1
    max_parallel_tasks: int = 1
    seed: int = SEED
    control_mode: str = "relative"
    suite: str = "libero_goal"

    def __post_init__(self) -> None:
        if self.dev_episodes_per_task <= 0 or self.final_episodes_per_task <= 0:
            raise ValueError("evaluation episode counts must be positive")
        if self.eval_batch_size <= 0 or self.max_parallel_tasks <= 0:
            raise ValueError("evaluation parallelism must be positive")
        if self.control_mode not in {"relative", "absolute"}:
            raise ValueError("control_mode must be relative or absolute")


@dataclass(frozen=True)
class ExperimentProtocol:
    """The decisions that must not vary across comparable runs."""

    protocol_version: str = PROTOCOL_VERSION
    dataset_repo: str = DATASET_REPO
    dataset_revision: str = DATASET_REVISION
    model_repo: str = MODEL_REPO
    model_revision: str = MODEL_REVISION
    lerobot_version: str = LEROBOT_VERSION
    libero_assets_repo: str = LIBERO_ASSETS_REPO
    libero_assets_revision: str = LIBERO_ASSETS_REVISION
    seed: int = SEED
    validation_fraction: float = 0.20
    subset_fractions: tuple[float, ...] = (0.20, 0.50, 1.0)
    training_steps: int = 10_000
    checkpoint_interval: int = 2_000
    max_data_workers: int = 0
    evaluation_initializations: str = "manifests/eval_initializations.json"
    task_keys: tuple[str, ...] = (
        "goal_bowl_plate",
        "goal_bowl_stove",
        "goal_bowl_cabinet_top",
    )
    evaluation: EvalProtocol = field(default_factory=EvalProtocol)

    def __post_init__(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"unexpected protocol version: {self.protocol_version}")
        if not self.dataset_revision or not self.model_revision or not self.libero_assets_revision:
            raise ValueError("dataset, model, and LIBERO asset revisions must be pinned")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0, 1)")
        if self.subset_fractions != tuple(sorted(set(self.subset_fractions))):
            raise ValueError("subset_fractions must be sorted and unique")
        if self.subset_fractions[-1] != 1.0:
            raise ValueError("the largest data condition must use the complete train pool")
        if not self.evaluation_initializations:
            raise ValueError("evaluation_initializations must identify the frozen init manifest")
        if self.training_steps <= 0 or self.checkpoint_interval <= 0:
            raise ValueError("training budgets must be positive")
        if self.max_data_workers < 0:
            raise ValueError("max_data_workers cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["subset_fractions"] = list(self.subset_fractions)
        value["task_keys"] = list(self.task_keys)
        return value


def load_json(path: str | Path) -> Any:
    """Load JSON with a useful path in errors."""
    source = Path(path)
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"JSON file does not exist: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {source}: {exc}") from exc


def dump_json(value: Any, path: str | Path) -> None:
    """Write deterministic, human-readable JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a Wilson score interval for a binomial proportion."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("successes/trials must satisfy 0 <= successes <= trials")
    if trials == 0:
        return (float("nan"), float("nan"))
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _run_capture(command: list[str], timeout_s: float = 5.0) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s, check=False)
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc), "command": command}
    return {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "command": command,
    }


def collect_environment() -> dict[str, Any]:
    """Collect read-only host facts for ``docs/ENVIRONMENT.md`` and run records."""
    import shutil
    import sys

    facts: dict[str, Any] = {
        "captured_unix": time.time(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": __import__("os").cpu_count(),
        "executables": {name: shutil.which(name) for name in ("python", "uv", "conda", "lerobot-train", "lerobot-eval")},
    }
    facts["nvidia_smi"] = _run_capture(["nvidia-smi"])
    facts["mujoco_gl"] = __import__("os").environ.get("MUJOCO_GL")
    try:
        import resource

        facts["process_memory_limit_bytes"] = resource.getrlimit(resource.RLIMIT_AS)[0]
    except (ImportError, AttributeError):
        facts["process_memory_limit_bytes"] = None
    return facts


def require_unique(values: Iterable[int], label: str) -> list[int]:
    result = [int(value) for value in values]
    if len(result) != len(set(result)):
        raise ValueError(f"{label} contains duplicate IDs")
    if any(value < 0 for value in result):
        raise ValueError(f"{label} contains a negative ID")
    return result


def validate_disjoint(groups: Mapping[str, Iterable[int]]) -> None:
    seen: dict[int, str] = {}
    for group_name, values in groups.items():
        for value in require_unique(values, group_name):
            if value in seen:
                raise ValueError(f"episode {value} occurs in both {seen[value]} and {group_name}")
            seen[value] = group_name
