#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# WSLg needs the D3D12 Gallium driver explicitly; robosuite overrides glfw to egl.
export MUJOCO_GL=glx
export GALLIUM_DRIVER=d3d12
export MESA_LOADER_DRIVER_OVERRIDE=d3d12
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
exec .venv/bin/python -m sim_vla.latency_v2.cli "$@"
