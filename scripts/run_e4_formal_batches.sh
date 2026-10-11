#!/usr/bin/env bash
# Run E4 formal evaluation as four independently recoverable five-episode batches.
# The batches use disjoint seed/initialization ranges and are merged only after
# scripts/merge_e4_batches.py validates the protocol metadata.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="$ROOT_DIR/.venv/bin/python"
EVAL="$ROOT_DIR/.venv/bin/sim-vla-paired-language-eval"
CHECKPOINT="$ROOT_DIR/runs/train_100pct_formal/checkpoints/010000/pretrained_model"

for executable in "$PYTHON" "$EVAL"; do
  if [[ ! -x "$executable" ]]; then
    echo "required .venv entrypoint is missing: $executable" >&2
    exit 1
  fi
done
if [[ ! -d "$CHECKPOINT" ]]; then
  echo "missing formal checkpoint: $CHECKPOINT" >&2
  exit 1
fi

# Fail before creating a partial result if this WSL session has no usable CUDA
# device.  Formal SmolVLA evaluation is intentionally not silently downgraded
# to CPU because that changes runtime characteristics and is impractical here.
"$PYTHON" - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit(
        "CUDA device unavailable; restore the WSL/host NVIDIA device before starting E4 formal batches"
    )
print(f"CUDA preflight passed: {torch.cuda.get_device_name(0)}")
PY

export LIBERO_CONFIG_PATH="$ROOT_DIR/.cache/libero-config"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HOME="$ROOT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$ROOT_DIR/.cache/huggingface/datasets"
export HF_HUB_CACHE="$ROOT_DIR/.cache/huggingface/hub"
export MPLCONFIGDIR="$ROOT_DIR/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"
export PATH="$ROOT_DIR/.venv/bin:$PATH"

start_seeds=(52 57 62 67)
start_indices=(10 15 20 25)
for batch in 0 1 2 3; do
  batch_name="$(printf 'batch_%02d' "$batch")"
  output_root="$ROOT_DIR/results/e4_batches/$batch_name"
  output_dir="$output_root/e4_empty_formal"
  if [[ -s "$output_dir/eval_info.json" \
     && -s "$output_dir/protocol.json" \
     && -s "$output_dir/episode_records.json" \
     && -s "$output_dir/summary.json" ]]; then
    echo "E4 $batch_name already complete; skipping evaluator."
    continue
  fi
  mkdir -p "$output_root" "$ROOT_DIR/runs"
  log_path="$ROOT_DIR/runs/e4_${batch_name}.log"
  echo "$(date -Is) starting E4 $batch_name (seed ${start_seeds[$batch]}..$((start_seeds[$batch] + 4)), init ${start_indices[$batch]}..$((start_indices[$batch] + 4)))"
  env \
    LIBERO_CONFIG_PATH="$LIBERO_CONFIG_PATH" \
    MUJOCO_GL="$MUJOCO_GL" \
    MUJOCO_EGL_DEVICE_ID="$MUJOCO_EGL_DEVICE_ID" \
    HF_HUB_DISABLE_PROGRESS_BARS="$HF_HUB_DISABLE_PROGRESS_BARS" \
    HF_HOME="$HF_HOME" \
    HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
    HF_HUB_CACHE="$HF_HUB_CACHE" \
    "$EVAL" \
    --checkpoint "$CHECKPOINT" \
    --output-root "$output_root" \
    --phase formal \
    --episodes 5 \
    --seed 42 \
    --device cuda \
    --conditions e4 \
    --start-seed "${start_seeds[$batch]}" \
    --start-initialization-index "${start_indices[$batch]}" \
    --overwrite 2>&1 | tee "$log_path"
  for required in eval_info.json protocol.json episode_records.json summary.json; do
    if [[ ! -s "$output_dir/$required" ]]; then
      echo "E4 $batch_name ended without $output_dir/$required" >&2
      exit 1
    fi
  done
done

"$PYTHON" scripts/merge_e4_batches.py
"$PYTHON" - <<'PY'
import json
from pathlib import Path

root = Path("results")
status_path = root / "paired_language_status.json"
summary_path = root / "e4_empty_formal" / "summary.json"
batch_status_path = root / "e4_batches" / "batch_00" / "paired_language_status.json"
if not status_path.is_file() or not summary_path.is_file():
    raise SystemExit("cannot finalize paired-language status: missing root status or merged E4 summary")
status = json.loads(status_path.read_text(encoding="utf-8"))
summary = json.loads(summary_path.read_text(encoding="utf-8"))
formal = status.setdefault("conditions", {}).setdefault("formal", {})
formal.update(
    {
        "e4_output": "results/e4_empty_formal",
        "e4_summary": summary["overall"],
        "e4_status": "completed",
        "status": "completed",
    }
)
if batch_status_path.is_file():
    batch_status = json.loads(batch_status_path.read_text(encoding="utf-8"))
    probe = batch_status.get("shared_reset_probes", {}).get("formal")
    if probe is not None:
        status.setdefault("shared_reset_probes", {})["formal"] = probe
status["status"] = "completed"
status.pop("current_blocker", None)
status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
"$PYTHON" scripts/build_language_report.py
echo "$(date -Is) E4 formal batches and report completed."
