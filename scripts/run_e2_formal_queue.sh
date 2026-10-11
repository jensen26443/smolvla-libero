#!/usr/bin/env bash
# Queue the independent E2 data-budget runs behind the single-GPU E1 rollout.
# The script is fail-closed: it never removes an existing run, never overwrites
# a completed eval_info.json, and stops if a required final checkpoint is absent.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
QUEUE_LOG="$ROOT_DIR/runs/e2_formal_queue.log"
mkdir -p "$ROOT_DIR/runs"
exec > >(tee -a "$QUEUE_LOG") 2>&1

PYTHON="$ROOT_DIR/.venv/bin/python"
TRAIN="$ROOT_DIR/.venv/bin/lerobot-train"
EVAL="$ROOT_DIR/.venv/bin/lerobot-eval"
SUMMARIZE="$ROOT_DIR/.venv/bin/sim-vla-summarize-lerobot"
for executable in "$PYTHON" "$TRAIN" "$EVAL" "$SUMMARIZE"; do
  if [[ ! -x "$executable" ]]; then
    echo "required .venv entrypoint is missing: $executable" >&2
    exit 1
  fi
done

export LIBERO_CONFIG_PATH="$ROOT_DIR/.cache/libero-config"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HOME="$ROOT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$ROOT_DIR/.cache/huggingface/datasets"
export HF_HUB_CACHE="$ROOT_DIR/.cache/huggingface/hub"
export PATH="$ROOT_DIR/.venv/bin:$PATH"

set_e2_status() {
  local state="$1"
  local message="$2"
  STATUS_VALUE="$state" MESSAGE_VALUE="$message" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

path = Path("runs/experiment_status.json")
payload = json.loads(path.read_text(encoding="utf-8"))
condition = payload["conditions"]["E2"]
condition["status"] = os.environ["STATUS_VALUE"]
condition["queue_message"] = os.environ["MESSAGE_VALUE"]
condition["reason"] = os.environ["MESSAGE_VALUE"]
condition["formal_eval_status"] = (
    "completed" if os.environ["STATUS_VALUE"] == "formal_completed" else os.environ["STATUS_VALUE"]
)
payload["formal_protocol"]["status"] = "E1_completed_E2_in_progress"
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

latest_complete_checkpoint() {
  local output_dir="$1"
  "$PYTHON" - "$output_dir" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "checkpoints"
candidates = []
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
if candidates:
    print(max(candidates, key=lambda item: item[0])[1])
PY
}

write_resume_command() {
  local checkpoint_dir="$1"
  local command_file="$2"
  CHECKPOINT_DIR="$checkpoint_dir" COMMAND_FILE="$command_file" ROOT="$ROOT_DIR" "$PYTHON" - <<'PY'
import os
from pathlib import Path

checkpoint = Path(os.environ["CHECKPOINT_DIR"])
command_file = Path(os.environ["COMMAND_FILE"])
root = Path(os.environ["ROOT"])
command = (
    f'LIBERO_CONFIG_PATH="$PWD/.cache/libero-config" MUJOCO_GL=egl '
    f'MUJOCO_EGL_DEVICE_ID=0 HF_HUB_DISABLE_PROGRESS_BARS=1 '
    f'HF_HOME="$PWD/.cache/huggingface" '
    f'HF_DATASETS_CACHE="$PWD/.cache/huggingface/datasets" '
    f'HF_HUB_CACHE="$PWD/.cache/huggingface/hub" '
    f'{root}/.venv/bin/lerobot-train '
    f'--config_path={checkpoint / "pretrained_model" / "train_config.json"} '
    '--resume=true --steps=10000 --save_freq=2000 --eval_freq=20000'
)
command_file.parent.mkdir(parents=True, exist_ok=True)
command_file.write_text(command + "\n", encoding="utf-8")
print(command)
PY
}

wait_for_e1() {
  if [[ -s results/e1_formal/eval_info.json ]]; then
    echo "E1 eval_info.json already exists; continuing."
    return
  fi
  echo "Waiting for the active E1 official evaluator to finish..."
  while pgrep -f '[.]venv/bin/python .*lerobot-eval.*results/e1_formal' >/dev/null 2>&1 \
     || pgrep -f '[l]erobot-eval.*results/e1_formal' >/dev/null 2>&1; do
    sleep 60
  done
  if [[ ! -s results/e1_formal/eval_info.json ]]; then
    echo "E1 evaluator exited without results/e1_formal/eval_info.json" >&2
    exit 1
  fi
  echo "E1 official evaluator finished."
}

complete_e1_record() {
  if [[ ! -s results/e1_formal/summary.json ]]; then
    "$SUMMARIZE" results/e1_formal/eval_info.json \
      --condition E1 --final-start-index 10 \
      --output results/e1_formal/summary.json
  fi
  "$PYTHON" - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path

summary_path = Path("results/e1_formal/summary.json")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
overall = summary["overall"]["overall"]
status_path = Path("runs/experiment_status.json")
status = json.loads(status_path.read_text(encoding="utf-8"))
condition = status["conditions"]["E1"]
condition.update(
    {
        "status": "formal_completed",
        "formal_eval_status": "completed",
        "formal_run_status": "completed",
        "formal_summary": str(summary_path),
        "formal_successes": overall["successes"],
        "formal_trials": overall["trials"],
        "formal_success_rate": overall["success_rate"],
        "formal_wilson_95": overall["wilson_95"],
        "formal_task_results": {
            key: {"successes": value["successes"], "trials": value["trials"]}
            for key, value in summary["overall"]["by_task"].items()
        },
    }
)
status["formal_protocol"]["status"] = "E1_completed_E2_in_progress"
status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
run_path = Path("runs/e1_formal.run.json")
run = json.loads(run_path.read_text(encoding="utf-8"))
run.update(
    {
        "status": "completed",
        "finished_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "summary": str(summary_path),
        "overall_result": {
            "successes": overall["successes"],
            "trials": overall["trials"],
            "success_rate": overall["success_rate"],
        },
        "task_results": {
            key: {"successes": value["successes"], "trials": value["trials"]}
            for key, value in summary["overall"]["by_task"].items()
        },
        "note": "Formal statistics were generated only after all three task rollouts completed.",
    }
)
run_path.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

run_budget() {
  local budget="$1"
  local train_command="runs/train_$budget""_formal.command.txt"
  local output_dir="runs/train_$budget""_formal"
  local checkpoint="$output_dir/checkpoints/010000/pretrained_model"
  local eval_output="results/e2_$budget""_formal"
  local eval_command="runs/eval_e2_$budget""_formal.command.txt"
  local summary_output="$eval_output/summary.json"
  local run_record="runs/e2_$budget""_formal.run.json"

  if [[ ! -s "$train_command" ]]; then
    echo "missing command record: $train_command" >&2
    exit 1
  fi
  if [[ ! -d "$checkpoint" ]]; then
    resume_checkpoint="$(latest_complete_checkpoint "$output_dir")"
    if [[ -n "$resume_checkpoint" ]]; then
      resume_command_file="runs/train_$budget""_formal.resume.command.txt"
      write_resume_command "$resume_checkpoint" "$resume_command_file" >/dev/null
      set_e2_status "formal_training_$budget" "resuming independent $budget training from $resume_checkpoint"
      echo "Resuming independent E2 $budget training from $resume_checkpoint."
      bash -c "$(<"$resume_command_file")"
    else
      set_e2_status "formal_training_$budget" "independent $budget training started"
      echo "Starting independent E2 $budget training."
      bash -c "$(<"$train_command")"
    fi
  else
    echo "E2 $budget final checkpoint already exists; skipping training."
  fi
  if [[ ! -f "$checkpoint/config.json" || ! -f "$checkpoint/model.safetensors" ]]; then
    echo "E2 $budget ended without a complete final checkpoint: $checkpoint" >&2
    exit 1
  fi
  if [[ ! -s "$eval_output/eval_info.json" ]]; then
    if [[ ! -s "$eval_command" ]]; then
      echo "missing eval command record: $eval_command" >&2
      exit 1
    fi
    set_e2_status "formal_evaluation_$budget" "E2 $budget checkpoint ready; official rollout started"
    mkdir -p "$eval_output"
    bash -c "$(<"$eval_command")"
  else
    echo "E2 $budget eval_info.json already exists; skipping evaluator."
  fi
  if [[ ! -s "$eval_output/eval_info.json" ]]; then
    echo "E2 $budget evaluator ended without eval_info.json" >&2
    exit 1
  fi
  "$SUMMARIZE" "$eval_output/eval_info.json" \
    --condition "E2_$budget" --final-start-index 10 \
    --output "$summary_output"
  "$PYTHON" - "$budget" "$summary_output" "$run_record" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

budget, summary_name, record_name = sys.argv[1:]
summary = json.loads(Path(summary_name).read_text(encoding="utf-8"))
overall = summary["overall"]["overall"]
record = {
    "condition": f"E2_{budget}",
    "budget": budget,
    "status": "completed",
    "summary": summary_name,
    "finished_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    "protocol_version": "sim-vla-libero-v1",
    "episodes_per_task_total": 30,
    "development_episodes_per_task": 10,
    "final_episodes_per_task": 20,
    "seed": 42,
    "checkpoint": f"runs/train_{budget}_formal/checkpoints/010000/pretrained_model",
    "task_results": {
        key: {"successes": value["successes"], "trials": value["trials"]}
        for key, value in summary["overall"]["by_task"].items()
    },
    "overall_result": {
        "successes": overall["successes"],
        "trials": overall["trials"],
        "success_rate": overall["success_rate"],
        "wilson_95": overall["wilson_95"],
    },
}
Path(record_name).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
status_path = Path("runs/experiment_status.json")
status = json.loads(status_path.read_text(encoding="utf-8"))
e2 = status["conditions"]["E2"]
e2.setdefault("formal_results", {})[budget] = record["overall_result"]
e2.setdefault("formal_task_results", {})[budget] = record["task_results"]
e2["formal_summary"] = {
    key: f"results/e2_{key}_formal/summary.json" for key in e2["formal_results"]
}
e2["status"] = "formal_training_or_evaluation_in_progress"
e2["formal_eval_status"] = "running"
status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

echo "$(date -Is) E2 queue started; log=$QUEUE_LOG"
wait_for_e1
complete_e1_record
run_budget 020pct
run_budget 050pct

# S4.3 permits reusing the identical E1 maximum-data checkpoint as E2 100%.
"$PYTHON" - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path

source = Path("results/e1_formal/summary.json")
summary = json.loads(source.read_text(encoding="utf-8"))
overall = summary["overall"]["overall"]
record = {
    "condition": "E2_100pct",
    "budget": "100pct",
    "status": "reused_from_E1",
    "source_condition": "E1",
    "source_checkpoint": "runs/train_100pct_formal/checkpoints/010000/pretrained_model",
    "source_summary": str(source),
    "finished_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    "protocol_version": "sim-vla-libero-v1",
    "episodes_per_task_total": 30,
    "development_episodes_per_task": 10,
    "final_episodes_per_task": 20,
    "seed": 42,
    "overall_result": {
        "successes": overall["successes"],
        "trials": overall["trials"],
        "success_rate": overall["success_rate"],
        "wilson_95": overall["wilson_95"],
    },
}
Path("runs/e2_100pct_formal.run.json").write_text(
    json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
status_path = Path("runs/experiment_status.json")
status = json.loads(status_path.read_text(encoding="utf-8"))
e2 = status["conditions"]["E2"]
e2.setdefault("formal_results", {})["100pct"] = record["overall_result"]
e2.setdefault("formal_task_results", {})["100pct"] = {
    key: {"successes": value["successes"], "trials": value["trials"]}
    for key, value in summary["overall"]["by_task"].items()
}
e2["status"] = "formal_completed"
e2["formal_eval_status"] = "completed"
e2["formal_run_status"] = "completed"
e2["formal_summary"] = {
    key: f"results/e2_{key}_formal/summary.json" for key in e2["formal_results"] if key != "100pct"
}
e2["formal_summary"]["100pct"] = "results/e1_formal/summary.json"
e2["reuse_note"] = "100pct reuses the identical E1 training/evaluation result per S4.3."
status["formal_protocol"]["status"] = "E0_E1_E2_completed_E3_E5_pending"
status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
echo "$(date -Is) E2 queue completed."
