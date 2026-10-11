"""Detach the authorized complete execution queue while retaining stage logs."""
import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-pid", type=int)
    parser.add_argument("--candidate-pid", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = root/"runs/latency_v2_pipeline"
    directory.mkdir(parents=True, exist_ok=True)
    command = ["bash", "scripts/run_latency_v2.sh", "pipeline"]
    if args.baseline_pid is not None:
        command += ["--baseline-pid", str(args.baseline_pid)]
    if args.candidate_pid is not None:
        command += ["--candidate-pid", str(args.candidate_pid)]
    if args.resume:
        command.append("--resume")
    with (directory/"queue.log").open("a") as log:
        process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({"pid": process.pid, "log": str(directory/"queue.log"), "command": command}))


if __name__ == "__main__":
    main()
