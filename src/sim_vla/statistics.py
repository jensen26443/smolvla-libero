"""Evaluation records and conservative success-rate summaries."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .protocol import load_json, wilson_interval


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str) and value.lower() in {"true", "1", "yes", "success"}:
        return True
    if isinstance(value, str) and value.lower() in {"false", "0", "no", "failure"}:
        return False
    raise ValueError(f"cannot interpret success value {value!r}")


def load_trial_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {source}:{line_number}: {exc}") from exc
        if not isinstance(value, Mapping):
            raise TypeError(f"trial record at line {line_number} is not an object")
        records.append(dict(value))
    return records


def load_lerobot_eval_records(
    path: str | Path,
    *,
    condition: str = "unspecified",
    task_key_by_id: Mapping[int | str, str] | None = None,
    final_start_index: int | None = None,
) -> list[dict[str, Any]]:
    """Convert official LeRobot ``eval_info.json`` arrays to trial records.

    LeRobot 0.5.1 stores per-task reward/success arrays in ``per_task`` while
    its top-level output omits the per-episode seed.  The conversion preserves
    task and local episode order, marks every emitted row valid, and optionally
    records the reserved final-slice boundary used by this protocol.  It does
    not infer a physical failure cause beyond the official success predicate.
    """

    payload = load_json(path)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("per_task"), list):
        raise TypeError(f"not an official LeRobot eval_info.json: {path}")
    if final_start_index is not None and final_start_index < 0:
        raise ValueError("final_start_index must be non-negative")
    lookup = {str(key): value for key, value in (task_key_by_id or {}).items()}
    records: list[dict[str, Any]] = []
    for task_result in payload["per_task"]:
        if not isinstance(task_result, Mapping):
            raise TypeError("per_task entries must be objects")
        task_group = str(task_result.get("task_group", "unspecified"))
        task_id = task_result.get("task_id")
        task_key = lookup.get(str(task_id), f"{task_group}_{task_id}")
        metrics = task_result.get("metrics")
        if not isinstance(metrics, Mapping):
            raise TypeError("per_task entry is missing metrics")
        successes = metrics.get("successes", [])
        sum_rewards = metrics.get("sum_rewards", [])
        max_rewards = metrics.get("max_rewards", [])
        if not isinstance(successes, list) or not isinstance(sum_rewards, list) or not isinstance(max_rewards, list):
            raise TypeError("LeRobot metrics arrays must be lists")
        if not len(successes) == len(sum_rewards) == len(max_rewards):
            raise ValueError(f"LeRobot metrics arrays have inconsistent lengths for task {task_id}")
        for episode_index, (success, sum_reward, max_reward) in enumerate(
            zip(successes, sum_rewards, max_rewards, strict=True)
        ):
            row: dict[str, Any] = {
                "condition": condition,
                "task_group": task_group,
                "task_id": task_id,
                "task_key": task_key,
                "episode_index": episode_index,
                "success": _as_bool(success),
                "sum_reward": float(sum_reward),
                "max_reward": float(max_reward),
                "valid_trial": True,
            }
            if final_start_index is not None:
                row["slice"] = "final" if episode_index >= final_start_index else "development"
            if not row["success"]:
                row["failure_type"] = "official_success_predicate_false"
            records.append(row)
    return records


def summarize_lerobot_eval(
    path: str | Path,
    *,
    condition: str = "unspecified",
    task_key_by_id: Mapping[int | str, str] | None = None,
    final_start_index: int | None = None,
) -> dict[str, Any]:
    """Return the standard Wilson/failure summary for an official eval file."""

    records = load_lerobot_eval_records(
        path,
        condition=condition,
        task_key_by_id=task_key_by_id,
        final_start_index=final_start_index,
    )
    if final_start_index is not None:
        slices = {
            "development": [record for record in records if record.get("slice") == "development"],
            "final": [record for record in records if record.get("slice") == "final"],
        }
        return {
            "source": str(path),
            "records": len(records),
            "overall": summarize_trials(records),
            "slices": {name: summarize_trials(values) for name, values in slices.items()},
        }
    return {"source": str(path), "records": len(records), "overall": summarize_trials(records)}


def _summary(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if bool(record.get("valid_trial", True))]
    invalid = [record for record in records if not bool(record.get("valid_trial", True))]
    successes = sum(_as_bool(record.get("success", False)) for record in valid)
    trials = len(valid)
    low, high = wilson_interval(successes, trials)
    failures = Counter(str(record.get("failure_type", "unspecified")) for record in valid if not _as_bool(record.get("success", False)))
    return {
        "successes": successes,
        "trials": trials,
        "success_rate": successes / trials if trials else None,
        "wilson_95": [low, high] if trials else None,
        "invalid_trials": len(invalid),
        "failure_types": dict(sorted(failures.items())),
    }


def summarize_trials(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Group valid and invalid trials without silently changing denominators."""
    all_records = list(records)
    by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_instruction: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in all_records:
        by_condition[str(record.get("condition", "unspecified"))].append(record)
        by_task[str(record.get("task_key", record.get("task", "unspecified")))].append(record)
        if "instruction_id" in record:
            by_instruction[str(record["instruction_id"])].append(record)
    return {
        "total_records": len(all_records),
        "overall": _summary(all_records),
        "by_condition": {key: _summary(values) for key, values in sorted(by_condition.items())},
        "by_task": {key: _summary(values) for key, values in sorted(by_task.items())},
        "by_instruction": {key: _summary(values) for key, values in sorted(by_instruction.items())},
    }
