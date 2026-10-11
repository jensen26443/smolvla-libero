"""Leakage-resistant episode manifests and deterministic data-efficiency subsets."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .protocol import ExperimentProtocol, dump_json, require_unique, validate_disjoint
from .tasks import TASK_BY_DATASET_INDEX, TASK_BY_KEY, TaskSpec


def _coerce_task_key(value: Any) -> str:
    if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
        task = TASK_BY_DATASET_INDEX.get(int(value))
        if task is None:
            raise ValueError(f"dataset task index {value} is not one of the selected LIBERO tasks")
        return task.key
    value = str(value)
    if value in TASK_BY_KEY:
        return value
    by_name = {task.task_name: task.key for task in TASK_BY_KEY.values()}
    by_name.update({task.instruction.lower(): task.key for task in TASK_BY_KEY.values()})
    by_name.update({task.task_name.replace("_", " "): task.key for task in TASK_BY_KEY.values()})
    if value in by_name:
        return by_name[value]
    raise ValueError(f"unknown task value in episode index: {value!r}")


def load_episode_index(path: str | Path, selected: Iterable[TaskSpec] | None = None) -> dict[str, list[int]]:
    """Load episode-to-task rows from JSON/JSONL/CSV or a small parquet file.

    Accepted JSON forms are a list of rows, ``{"episodes": [...]}``, or a
    mapping from task key/name/index to episode IDs. Parquet support is optional
    and uses pyarrow only when present, so protocol tests stay lightweight.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    selected_keys = {task.key for task in selected} if selected is not None else set(TASK_BY_KEY)

    if source.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("reading parquet requires pyarrow; export metadata to JSON first") from exc
        rows = pq.read_table(source, columns=["episode_index", "tasks"]).to_pylist()
    elif source.suffix.lower() == ".csv":
        with source.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    elif source.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping) and "episodes_by_task" in payload:
            payload = payload["episodes_by_task"]
        if isinstance(payload, Mapping) and "episodes" in payload:
            rows = payload["episodes"]
        elif isinstance(payload, list):
            rows = payload
        elif isinstance(payload, Mapping):
            result: dict[str, list[int]] = defaultdict(list)
            for task_value, episode_values in payload.items():
                key = _coerce_task_key(task_value)
                if key in selected_keys:
                    result[key].extend(int(item) for item in episode_values)
            return _validate_index(result, selected_keys)
        else:
            raise ValueError(f"unsupported episode index JSON shape in {source}")

    result = defaultdict(list)
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError(f"episode row must be an object, got {type(row).__name__}")
        episode_value = row.get("episode_index", row.get("episode_id"))
        if episode_value is None:
            raise ValueError("episode row is missing episode_index")
        task_value = row.get("task_key", row.get("task_index", row.get("task_id", row.get("task"))))
        if task_value is None:
            tasks = row.get("tasks")
            task_value = tasks[0] if isinstance(tasks, list) and tasks else tasks
        key = _coerce_task_key(task_value)
        if key in selected_keys:
            result[key].append(int(episode_value))
    return _validate_index(result, selected_keys)


def _validate_index(index: Mapping[str, Iterable[int]], selected_keys: set[str]) -> dict[str, list[int]]:
    result = {key: sorted(set(require_unique(values, key))) for key, values in index.items() if key in selected_keys}
    missing = sorted(selected_keys - set(result))
    if missing:
        raise ValueError(f"episode index has no rows for selected task(s): {missing}")
    flattened = [episode for values in result.values() for episode in values]
    if len(flattened) != len(set(flattened)):
        raise ValueError("an episode is assigned to more than one selected task")
    return result


def make_nested_manifests(
    episodes_by_task: Mapping[str, Iterable[int]],
    protocol: ExperimentProtocol | None = None,
    source: str = "unknown",
) -> dict[str, dict[str, Any]]:
    """Create train/validation manifests for all nested data budgets."""
    protocol = protocol or ExperimentProtocol()
    selected = list(protocol.task_keys)
    normalized = {key: sorted(set(require_unique(episodes_by_task.get(key, []), key))) for key in selected}
    if any(not normalized[key] for key in selected):
        missing = [key for key in selected if not normalized[key]]
        raise ValueError(f"no episodes for task(s): {missing}")

    per_task: dict[str, dict[str, list[int]]] = {}
    for key in selected:
        shuffled = list(normalized[key])
        random.Random(protocol.seed + selected.index(key)).shuffle(shuffled)
        validation_count = max(1, int(len(shuffled) * protocol.validation_fraction + 0.999999))
        if validation_count >= len(shuffled):
            raise ValueError(f"task {key} needs at least two episodes for a train/validation split")
        per_task[key] = {
            "validation": sorted(shuffled[-validation_count:]),
            "train_pool": shuffled[:-validation_count],
        }

    all_manifests: dict[str, dict[str, Any]] = {}
    for fraction in protocol.subset_fractions:
        name = f"train_{int(fraction * 100):03d}pct"
        train: dict[str, list[int]] = {}
        validation: dict[str, list[int]] = {}
        for key in selected:
            pool = per_task[key]["train_pool"]
            count = max(1, int(len(pool) * fraction))
            train[key] = sorted(pool[:count])
            validation[key] = per_task[key]["validation"]
        validate_disjoint({f"{name}.train.{key}": values for key, values in train.items()} | {
            f"{name}.validation.{key}": values for key, values in validation.items()
        })
        all_manifests[name] = {
            "protocol_version": protocol.protocol_version,
            "source_dataset": source,
            "seed": protocol.seed,
            "fraction_of_train_pool": fraction,
            "task_keys": selected,
            "train_episodes": train,
            "validation_episodes": validation,
            "counts": {
                "train": {key: len(values) for key, values in train.items()},
                "validation": {key: len(values) for key, values in validation.items()},
            },
        }
    _validate_nested(all_manifests)
    return all_manifests


def _validate_nested(manifests: Mapping[str, Mapping[str, Any]]) -> None:
    ordered = sorted(manifests.values(), key=lambda item: float(item["fraction_of_train_pool"]))
    previous: dict[str, set[int]] | None = None
    for manifest in ordered:
        train = {key: set(values) for key, values in manifest["train_episodes"].items()}
        validation = {key: set(values) for key, values in manifest["validation_episodes"].items()}
        for key, train_values in train.items():
            if train_values & validation[key]:
                raise ValueError(f"train/validation overlap in {key}")
        current = {key: values for key, values in train.items()}
        if previous is not None:
            for key, current_values in current.items():
                if not previous[key] <= current_values:
                    raise ValueError("data-efficiency subsets are not nested")
        previous = current


def write_manifests(manifests: Mapping[str, Mapping[str, Any]], output_dir: str | Path) -> list[Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, manifest in manifests.items():
        path = target / f"{name}.json"
        dump_json(manifest, path)
        paths.append(path)
    return paths
