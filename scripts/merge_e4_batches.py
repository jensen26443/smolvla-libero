#!/usr/bin/env python
"""Merge crash-resistant E4 formal batches into the canonical result folder.

Each batch is a five-episode invocation of ``paired-language-eval`` with a
disjoint seed and shared-initialization-index range.  This utility validates
the ranges before concatenating the official LeRobot metrics and episode log;
it never reruns simulation.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from sim_vla.paired_language_eval import _summary_from_eval
from sim_vla.protocol import dump_json

ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _merge_infos(infos: list[dict[str, Any]]) -> dict[str, Any]:
    if not infos:
        raise ValueError("no E4 batch eval_info files")
    first_tasks = infos[0].get("per_task", [])
    if not isinstance(first_tasks, list) or len(first_tasks) != 1:
        raise ValueError("E4 batches must contain exactly one per-task result")
    task_id = int(first_tasks[0]["task_id"])
    task_group = str(first_tasks[0]["task_group"])
    sums: list[float] = []
    maximums: list[float] = []
    successes: list[bool] = []
    video_paths: list[str] = []
    eval_seconds = 0.0
    for info in infos:
        per_task = info.get("per_task", [])
        if not isinstance(per_task, list) or len(per_task) != 1:
            raise ValueError("each E4 batch must contain exactly one per-task result")
        task = per_task[0]
        if int(task["task_id"]) != task_id or str(task["task_group"]) != task_group:
            raise ValueError("E4 batch task identity mismatch")
        metrics = task.get("metrics", {})
        for key, destination in (
            ("sum_rewards", sums),
            ("max_rewards", maximums),
            ("successes", successes),
        ):
            values = metrics.get(key, [])
            if not isinstance(values, list):
                raise TypeError(f"invalid {key} in E4 batch")
            destination.extend(values)
        paths = metrics.get("video_paths", [])
        if isinstance(paths, list):
            video_paths.extend(str(path) for path in paths)
        eval_seconds += float(info.get("overall", {}).get("eval_s", 0.0))
    if not (len(sums) == len(maximums) == len(successes)):
        raise ValueError("E4 metric arrays have inconsistent lengths")
    if not successes:
        raise ValueError("E4 batches contain no episodes")
    episodes = len(successes)
    avg_sum = statistics.fmean(float(value) for value in sums)
    avg_max = statistics.fmean(float(value) for value in maximums)
    success_rate = statistics.fmean(bool(value) for value in successes) * 100.0
    return {
        "per_task": [
            {
                "task_group": task_group,
                "task_id": task_id,
                "metrics": {
                    "sum_rewards": sums,
                    "max_rewards": maximums,
                    "successes": [bool(value) for value in successes],
                    "video_paths": video_paths,
                },
            }
        ],
        "per_group": {
            task_group: {
                "avg_sum_reward": avg_sum,
                "avg_max_reward": avg_max,
                "pc_success": success_rate,
                "n_episodes": episodes,
                "video_paths": video_paths,
            }
        },
        "overall": {
            "avg_sum_reward": avg_sum,
            "avg_max_reward": avg_max,
            "pc_success": success_rate,
            "n_episodes": episodes,
            "eval_s": eval_seconds,
            "eval_ep_s": eval_seconds / episodes,
            "video_paths": video_paths,
        },
    }


def merge(*, batches_root: Path, output_dir: Path, expected_batches: int = 4) -> dict[str, Any]:
    batch_dirs = sorted(path for path in batches_root.glob("batch_*") if path.is_dir())
    if len(batch_dirs) != expected_batches:
        raise FileNotFoundError(
            f"expected {expected_batches} E4 batch directories under {batches_root}, found {len(batch_dirs)}"
        )
    entries: list[tuple[int, int, Path, dict[str, Any], dict[str, Any], list[dict[str, Any]]]] = []
    for batch_dir in batch_dirs:
        directory = batch_dir / "e4_empty_formal"
        protocol = _read(directory / "protocol.json")
        info = _read(directory / "eval_info.json")
        records_payload = _read(directory / "episode_records.json")
        records = records_payload.get("records", [])
        if not isinstance(records, list):
            raise TypeError(f"invalid records in {directory}")
        start_seed = int(protocol["start_seed"])
        start_index = int(protocol["start_initialization_index"])
        episodes = int(protocol["episodes_per_task"])
        if episodes != len(records):
            raise ValueError(f"record count mismatch in {directory}: {len(records)} != {episodes}")
        protocol_seeds = [int(value) for value in protocol.get("seeds", [])]
        protocol_indices = [int(value) for value in protocol.get("initialization_indices", [])]
        if protocol_seeds != list(range(start_seed, start_seed + episodes)):
            raise ValueError(f"seed range mismatch in {directory}")
        if protocol_indices != list(range(start_index, start_index + episodes)):
            raise ValueError(f"initialization range mismatch in {directory}")
        metrics = info.get("per_task", [{}])[0].get("metrics", {})
        metric_lengths = {
            len(values) for key in ("sum_rewards", "max_rewards", "successes") if (values := metrics.get(key, []))
        }
        if metric_lengths != {episodes}:
            raise ValueError(f"metric count mismatch in {directory}: {metric_lengths} != {{{episodes}}}")
        if any(str(record.get("task_key")) != "goal_bowl_plate" for record in records):
            raise ValueError(f"unexpected task key in {directory}")
        entries.append((start_seed, start_index, batch_dir, protocol, info, records))
    entries.sort(key=lambda item: item[0])
    all_infos: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    all_seeds: list[int] = []
    all_indices: list[int] = []
    anchor = str(entries[0][3].get("shared_init_anchor", ""))
    state_count = int(entries[0][3].get("shared_init_state_count", 0))
    state_shape = entries[0][3].get("shared_init_state_shape", [])
    for offset, (start_seed, start_index, batch_dir, protocol, info, records) in enumerate(entries):
        expected_seed = 52 + offset * 5
        expected_index = 10 + offset * 5
        if start_seed != expected_seed or start_index != expected_index:
            raise ValueError(
                f"non-contiguous E4 batch {batch_dir.name}: seed/index {start_seed}/{start_index}; "
                f"expected {expected_seed}/{expected_index}"
            )
        if protocol.get("condition") != "E4_empty" or protocol.get("phase") != "formal":
            raise ValueError(f"unexpected protocol identity in {batch_dir}")
        if str(protocol.get("shared_init_anchor", "")) != anchor:
            raise ValueError("shared initialization anchor mismatch")
        if int(protocol.get("shared_init_state_count", 0)) != state_count:
            raise ValueError("shared initialization count mismatch")
        if protocol.get("shared_init_state_shape", []) != state_shape:
            raise ValueError("shared initialization shape mismatch")
        all_infos.append(info)
        all_records.extend(records)
        all_seeds.extend(int(value) for value in protocol["seeds"])
        all_indices.extend(int(value) for value in protocol["initialization_indices"])
    if len(all_records) != 20 or len(all_seeds) != 20 or len(all_indices) != 20:
        raise ValueError("merged E4 formal result must contain exactly 20 episodes")
    merged_info = _merge_infos(all_infos)
    task_map = {int(merged_info["per_task"][0]["task_id"]): "goal_bowl_plate"}
    summary = _summary_from_eval(merged_info, task_map, "E4_empty_formal")
    first_protocol = entries[0][3]
    merged_protocol = dict(first_protocol)
    merged_protocol.update(
        {
            "episodes_per_task": 20,
            "start_seed": 52,
            "seeds": all_seeds,
            "start_initialization_index": 10,
            "initialization_indices": all_indices,
            "batch_sources": [str(item[2].relative_to(ROOT)) for item in entries],
            "merge": "four validated five-episode batches; no simulation rerun during merge",
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(merged_info, output_dir / "eval_info.json")
    dump_json(merged_protocol, output_dir / "protocol.json")
    dump_json({"records": all_records, "count": len(all_records)}, output_dir / "episode_records.json")
    dump_json(summary, output_dir / "summary.json")
    result = {
        "status": "completed",
        "episodes": len(all_records),
        "successes": int(summary["overall"]["successes"]),
        "trials": int(summary["overall"]["trials"]),
        "success_rate": float(summary["overall"]["success_rate"]),
        "output": str(output_dir.relative_to(ROOT)),
        "batches": [str(item[2].relative_to(ROOT)) for item in entries],
    }
    dump_json(result, batches_root / "merge_status.json")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches-root", type=Path, default=ROOT / "results" / "e4_batches")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "e4_empty_formal")
    parser.add_argument("--expected-batches", type=int, default=4)
    args = parser.parse_args(argv)
    result = merge(batches_root=args.batches_root, output_dir=args.output_dir, expected_batches=args.expected_batches)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
