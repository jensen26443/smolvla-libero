#!/usr/bin/env bash
# Run the frozen E5 paraphrase probe and formal evaluation after E2.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
QUEUE_LOG="$ROOT_DIR/runs/e5_formal_queue.log"
mkdir -p "$ROOT_DIR/runs"
exec > >(tee -a "$QUEUE_LOG") 2>&1

PYTHON="$ROOT_DIR/.venv/bin/python"
SUMMARIZE="$ROOT_DIR/.venv/bin/sim-vla-summarize-lerobot"
for executable in "$PYTHON" "$SUMMARIZE"; do
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

set_e5_status() {
  local state="$1"
  local message="$2"
  STATUS_VALUE="$state" MESSAGE_VALUE="$message" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

path = Path("runs/experiment_status.json")
payload = json.loads(path.read_text(encoding="utf-8"))
condition = payload["conditions"]["E5"]
condition["status"] = os.environ["STATUS_VALUE"]
condition["queue_message"] = os.environ["MESSAGE_VALUE"]
condition["formal_eval_status"] = os.environ["STATUS_VALUE"]
payload["formal_protocol"]["status"] = "E2_completed_E5_in_progress"
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

mark_e5_waiting() {
  MESSAGE_VALUE="$1" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

path = Path("runs/experiment_status.json")
payload = json.loads(path.read_text(encoding="utf-8"))
condition = payload["conditions"]["E5"]
condition["status"] = "queued_after_e2"
condition["queue_message"] = os.environ["MESSAGE_VALUE"]
condition["reason"] = os.environ["MESSAGE_VALUE"]
condition["formal_eval_status"] = "waiting_for_e2"
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

wait_for_e2() {
  local message="waiting for E2 020pct, 050pct, and 100pct records; paraphrase input is frozen"
  mark_e5_waiting "$message"
  echo "Waiting for E2 020pct, 050pct, and 100pct records..."
  while [[ ! -s runs/e2_020pct_formal.run.json \
        || ! -s runs/e2_050pct_formal.run.json \
        || ! -s runs/e2_100pct_formal.run.json ]]; do
    if ! pgrep -f '[s]cripts/run_e2_formal_queue.sh' >/dev/null 2>&1; then
      echo "E2 queue is no longer running and its three records are incomplete" >&2
      exit 1
    fi
    sleep 60
  done
}

run_one() {
  local phase="$1"
  local output_dir="$2"
  local episodes="$3"
  local summary="$output_dir/summary.json"
  local eval_info="$output_dir/eval_info.json"
  local command_record
  if [[ "$phase" == "dev_probe" ]]; then
    command_record="runs/eval_e5_dev_probe.command.txt"
  else
    command_record="runs/eval_e5_formal.command.txt"
  fi

  if [[ ! -s "$eval_info" ]]; then
    if [[ "$phase" == "formal" ]]; then
      set_e5_status "formal_evaluation_formal" "E5 formal rollout started"
    else
      set_e5_status "development_probe" "E5 development probe rollout started"
    fi
    mkdir -p "$output_dir"
    if [[ ! -s "$command_record" ]]; then
      echo "missing E5 command record: $command_record" >&2
      exit 1
    fi
    bash -c "$(<"$command_record")"
  else
    echo "E5 $phase eval_info.json already exists; skipping evaluator."
  fi
  if [[ ! -s "$eval_info" ]]; then
    echo "E5 $phase evaluator ended without $eval_info" >&2
    exit 1
  fi
  if [[ "$phase" == "formal" ]]; then
    "$SUMMARIZE" "$eval_info" --condition "E5_$phase" \
      --final-start-index 10 --output "$summary"
  else
    "$SUMMARIZE" "$eval_info" --condition "E5_$phase" --output "$summary"
  fi
}

echo "$(date -Is) E5 queue started; log=$QUEUE_LOG"
wait_for_e2

if [[ ! -s results/e5_dev_probe/eval_info.json ]]; then
  set_e5_status "development_probe" "E5 one-episode-per-task language override probe started"
  run_one dev_probe results/e5_dev_probe 1
fi

run_one formal results/e5_formal 30

"$PYTHON" - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path

summary_path = Path("results/e5_formal/summary.json")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
overall = summary["overall"]["overall"]
record = {
    "condition": "E5",
    "status": "completed",
    "finished_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    "protocol_version": "sim-vla-libero-v1",
    "checkpoint": "runs/train_100pct_formal/checkpoints/010000/pretrained_model",
    "paraphrase_manifest": "manifests/semantic_paraphrases.json",
    "summary": str(summary_path),
    "episodes_per_task_total": 30,
    "development_episodes_per_task": 10,
    "final_episodes_per_task": 20,
    "seed": 42,
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
Path("runs/e5_formal.run.json").write_text(
    json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
status_path = Path("runs/experiment_status.json")
status = json.loads(status_path.read_text(encoding="utf-8"))
condition = status["conditions"]["E5"]
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
        "formal_task_results": record["task_results"],
    }
)
status["formal_protocol"]["status"] = "E0_E1_E2_E3_E4_custom_complete_E5_completed"
status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
"$PYTHON" scripts/build_report_artifacts.py --strict
echo "$(date -Is) E5 queue completed."
