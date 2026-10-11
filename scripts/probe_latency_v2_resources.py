"""Native versus projected GPU costs on one fixed development observation."""
import gc
import json
import time
from pathlib import Path

import numpy as np

from sim_vla.assets import project_root
from sim_vla.latency_v2.env import ControlledEnv
from sim_vla.latency_v2.experiments import CHECKPOINT
from sim_vla.latency_v2.integrity import file_sha256
from sim_vla.latency_v2.policy import PrefixPolicyAdapter
from sim_vla.latency_v2.schema import CommittedPrefix, InferenceRequest, ObservationSnapshot, digest_arrays
from sim_vla.latency_v2.storage import Budget, publish


def main():
    import torch

    torch.set_num_threads(4)
    root = project_root()
    output = root/"runs/latency_v2_resource_probe"
    if output.exists():
        raise FileExistsError("preserve prior resource measurement evidence")
    if (root/"runs/latency_v2_frozen/experiment_spec.json").exists():
        raise RuntimeError("this additional development resource audit must precede formal freeze")
    attempt = Budget(root/"runs/latency_v2_budget.json").claim("S3:resources:dev_init0", "S3_validation")
    publish(output/"attempt.json", {"attempt_number": attempt, "development_init": 0, "seed": 42,
                                  "rollout_scored": False})
    policy = PrefixPolicyAdapter(checkpoint=root/CHECKPOINT, warmup_calls=3)
    policy.warmup()
    env = ControlledEnv(task_key="goal_bowl_plate")
    try:
        observation = env.start_episode(0, 42)
        snapshot = ObservationSnapshot.capture("S3_resource_probe", 0, observation, env.instruction)
    finally:
        env.close()
    empty_actions = np.zeros((0, 7))
    empty = CommittedPrefix(0, empty_actions, (), digest_arrays(empty_actions))
    policy.reset()
    reference = policy.infer_request(InferenceRequest(attempt, 0, snapshot, 0, empty, 42, time.monotonic())).actions
    rows = []
    for length in (0, 5, 12, 22):
        actions = reference[:length].copy()
        prefix = CommittedPrefix(0, actions, (), digest_arrays(actions))
        samples = []
        for ordinal in range(5):
            policy.reset()
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            request = InferenceRequest(attempt, ordinal, snapshot, 0, prefix, 42, time.monotonic())
            started = time.monotonic()
            result = policy.infer_request(request)
            torch.cuda.synchronize()
            samples.append({"ordinal": ordinal, "endpoint_s": time.monotonic()-started,
                            "stages": result.stages, "allocated_before_bytes": before,
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                            "projection_max_error": max(result.sampler.get("projection_errors_before_after", [0.])),
                            "prefix_roundtrip_max_error": result.sampler.get("environment_prefix_roundtrip_max_error", 0.)})
        rows.append({"prefix_length": length, "sampler": "original" if length == 0 else "projected_clean_prefix",
                     "samples": samples,
                     "median_endpoint_s": float(np.median([row["endpoint_s"] for row in samples])),
                     "median_forward_s": float(np.median([row["stages"]["forward_s"] for row in samples])),
                     "max_peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in samples)})
    native = rows[0]
    for row in rows:
        row["median_endpoint_ratio_to_native"] = row["median_endpoint_s"]/native["median_endpoint_s"]
        row["additional_peak_allocated_bytes"] = row["max_peak_allocated_bytes"]-native["max_peak_allocated_bytes"]
    report = {"stage": "S3", "attempt_number": attempt, "environment_resets": 1,
              "formal_states_accessed": False, "rollout_scored": False,
              "fixed_development_observation_sha256": snapshot.digest, "seed": 42,
              "script_sha256": file_sha256(Path(__file__)),
              "calls": 21, "warmup_calls": 3, "rows": rows,
              "memory_scope": "PyTorch process allocated/reserved memory; excludes renderer, driver and desktop",
              "timing_scope": "five fixed-observation calls per prefix length; paired request ordinals; no task success evidence"}
    publish(output/"probe.json", report)
    if any(sample["projection_max_error"] > 0 or sample["prefix_roundtrip_max_error"] > 2e-5
           for row in rows for sample in row["samples"]):
        raise AssertionError("real resource probe failed the prefix implementation checks")
    publish(output/"complete.json", {"actual_gpu_measurement_complete": True})
    print(json.dumps({"rows": [{key: value for key, value in row.items() if key != "samples"} for row in rows]}))


if __name__ == "__main__":
    main()
