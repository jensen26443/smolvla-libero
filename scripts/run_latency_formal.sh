#!/usr/bin/env bash
# Resilient driver for the frozen formal matrix: re-invokes the CLI after
# infrastructure failures (WSL/GPU hiccups); the CLI itself resumes missing
# trials only. Stops when the matrix reports full completion.
set -u
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0

CONFIG="${1:-configs/latency/formal.yaml}"
MAX_RESTARTS="${2:-50}"

for attempt in $(seq 1 "$MAX_RESTARTS"); do
    echo "[run_latency_formal] attempt $attempt $(date -Is)" >> runs/latency_formal_driver.log
    .venv/bin/sim-vla-latency run --config "$CONFIG" --output-root runs \
        >> runs/latency_formal_v1.log 2>&1
    status=$?
    echo "[run_latency_formal] exit $status $(date -Is)" >> runs/latency_formal_driver.log
    if [ "$status" -eq 0 ]; then
        # CLI returns 0 even with infra-failed trials; check the state file.
        remaining=$(.venv/bin/python - <<'EOF'
import json
m = json.load(open("runs/latency_formal_v1/matrix.json"))
s = m["statuses"]
print(sum(1 for v in s.values() if v != "completed"))
EOF
)
        echo "[run_latency_formal] remaining=$remaining" >> runs/latency_formal_driver.log
        if [ "$remaining" -eq 0 ]; then
            echo "[run_latency_formal] matrix complete" >> runs/latency_formal_driver.log
            break
        fi
    fi
    sleep 60
done
