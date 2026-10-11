#!/usr/bin/env python
"""Record resumable state before a WSL/host restart.

This is intentionally a metadata-only operation.  It does not touch model
weights and does not send signals to running jobs; the caller can stop the
queues after this record has been written.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _latest_complete_checkpoint(output_dir: Path) -> tuple[int, Path] | None:
    root = output_dir / "checkpoints"
    candidates: list[tuple[int, Path]] = []
    if root.is_dir():
        for path in root.iterdir():
            if not path.is_dir() or not re.fullmatch(r"\d+", path.name):
                continue
            model = path / "pretrained_model"
            state = path / "training_state"
            if all((model / name).is_file() for name in ("config.json", "model.safetensors", "train_config.json")) and (
                state / "training_step.json"
            ).is_file():
                candidates.append((int(path.name), path))
    return max(candidates, key=lambda item: item[0]) if candidates else None


def _observed_step(log_path: Path) -> int | None:
    if not log_path.is_file():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    # tqdm writes carriage-return progress lines; the final matching line is
    # the best available estimate when the current step is not checkpointed.
    matches = re.findall(r"Training:\s+\d+%.*?\s(\d+)/\s*(\d+)\s+\[", text)
    if matches:
        return int(matches[-1][0])
    milestones = re.findall(r"step:(\d+)K", text)
    return int(milestones[-1]) * 1000 if milestones else None


def _processes() -> dict[str, list[int]]:
    try:
        output = subprocess.check_output(["ps", "-eo", "pid=,cmd="], text=True)
    except (OSError, subprocess.CalledProcessError):
        return {}
    found: dict[str, list[int]] = {"e2_queue": [], "e5_queue": [], "e2_050_train": []}
    for line in output.splitlines():
        match = re.match(r"\s*(\d+)\s+(.*)", line)
        if not match:
            continue
        pid, command = int(match.group(1)), match.group(2)
        if "run_e2_formal_queue.sh" in command:
            found["e2_queue"].append(pid)
        elif "run_e5_formal_queue.sh" in command:
            found["e5_queue"].append(pid)
        elif "train_050pct_formal" in command and "lerobot-train" in command:
            found["e2_050_train"].append(pid)
    return found


def _write_resume_command(checkpoint: Path) -> Path:
    command_path = ROOT / "runs" / "train_050pct_formal.resume.command.txt"
    command = (
        'LIBERO_CONFIG_PATH="$PWD/.cache/libero-config" MUJOCO_GL=egl '
        'MUJOCO_EGL_DEVICE_ID=0 HF_HUB_DISABLE_PROGRESS_BARS=1 '
        'HF_HOME="$PWD/.cache/huggingface" '
        'HF_DATASETS_CACHE="$PWD/.cache/huggingface/datasets" '
        'HF_HUB_CACHE="$PWD/.cache/huggingface/hub" '
        f'{ROOT}/.venv/bin/lerobot-train '
        f'--config_path={checkpoint / "pretrained_model" / "train_config.json"} '
        "--resume=true --steps=10000 --save_freq=2000 --eval_freq=20000"
    )
    command_path.write_text(command + "\n", encoding="utf-8")
    return command_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default="runs/e2_formal_queue.log")
    parser.add_argument("--output-dir", default="runs/train_050pct_formal")
    args = parser.parse_args(argv)

    output_dir = ROOT / args.output_dir
    log_path = ROOT / args.log
    complete = _latest_complete_checkpoint(output_dir)
    if complete is None:
        raise SystemExit(f"no complete checkpoint found under {output_dir / 'checkpoints'}")
    checkpoint_step, checkpoint = complete
    observed = _observed_step(log_path)
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    resume_path = _write_resume_command(checkpoint)
    processes = _processes()
    record: dict[str, Any] = {
        "protocol_version": "sim-vla-libero-v1",
        "condition": "E2_050pct",
        "status": "paused_for_wsl_restart",
        "paused_at": timestamp,
        "observed_training_step": observed,
        "latest_complete_checkpoint_step": checkpoint_step,
        "latest_complete_checkpoint": str(checkpoint.relative_to(ROOT)),
        "resume_command": str(resume_path.relative_to(ROOT)),
        "queue_scripts": {
            "e2": "scripts/run_e2_formal_queue.sh",
            "e5": "scripts/run_e5_formal_queue.sh",
        },
        "processes_at_save": processes,
        "uncheckpointed_updates": max(0, observed - checkpoint_step) if observed is not None else None,
        "post_restart": "restart E2 queue; it detects this checkpoint and resumes automatically",
    }
    pause_path = ROOT / "runs" / "e2_050pct_pause.json"
    pause_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    status_path = ROOT / "runs" / "experiment_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    e2 = status["conditions"]["E2"]
    e2["status"] = "formal_training_050pct"
    e2["formal_eval_status"] = "paused_for_wsl_restart"
    e2["queue_message"] = (
        f"paused for WSL restart at observed step {observed}; resuming from checkpoint {checkpoint_step}"
    )
    e2["reason"] = e2["queue_message"]
    e2["pause_record"] = str(pause_path.relative_to(ROOT))
    e2["resume_command"] = str(resume_path.relative_to(ROOT))
    e5 = status["conditions"]["E5"]
    e5["reason"] = "queue paused for WSL restart; paraphrase input is frozen"
    e5["queue_message"] = "stopped before WSL restart; restart the E2 queue first, then E5 will resume waiting"
    status["formal_protocol"]["status"] = "E1_completed_E2_paused_for_wsl_restart"
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
