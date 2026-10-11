"""Real E1 interface/clean-prefix validation, restricted to native development init 0."""
import json
import time

import numpy as np
import torch

from sim_vla.assets import project_root
from sim_vla.latency_v2.alignment import Executor
from sim_vla.latency_v2.env import ControlledEnv
from sim_vla.latency_v2.policy import PrefixPolicyAdapter
from sim_vla.latency_v2.schema import MethodConfig, ObservationSnapshot
from sim_vla.latency_v2.storage import Budget, publish


def main():
    root = project_root()
    directory = root / "runs/latency_v2_model_validation"
    if (directory / "validation.json").exists():
        raise FileExistsError("preserve completed real model validation")
    attempt = Budget(root/"runs/latency_v2_budget.json").claim("real-model-prefix-validation", "S3")
    publish(directory/"attempt.json", {"attempt_number": attempt, "native_init": 0, "seed": 42})
    torch.set_num_threads(4)
    env = ControlledEnv(task_key="goal_bowl_plate")
    try:
        obs = env.start_episode(0, 42)
        from OpenGL.GL import GL_RENDERER, glGetString

        renderer = glGetString(GL_RENDERER).decode()
        print("RENDERER", renderer, flush=True)
        snapshot = ObservationSnapshot.capture("native-dev-0", 0, obs, env.instruction)
        policy = PrefixPolicyAdapter(checkpoint=root/"runs/train_100pct_formal/checkpoints/010000/pretrained_model",
                                     warmup_calls=3, seed=42)
        policy.warmup()
        baseline = Executor(MethodConfig("B_RAW", 5), generation=1, seed=42, version="dev")
        request = baseline.submit(snapshot, 0)
        policy.reset()
        original = policy.infer(snapshot.packet(), 0, 0)
        policy.reset()
        zero = policy.infer_request(request).actions
        np.testing.assert_array_equal(original, zero)
        candidate = Executor(MethodConfig("C_PREFIX", 5, margin=2), generation=1, seed=42, version="dev")
        request = candidate.submit(snapshot, 5)
        policy.reset()
        projected = policy.infer_request(request)
        assert max(projected.sampler["projection_errors_before_after"]) == 0
        candidate.receive(projected, 5)
        for tick in range(7):
            candidate.advance(tick)
            action, _ = candidate.action(tick)
            np.testing.assert_array_equal(action, request.prefix.actions[tick])
            env.step(action)
        candidate.advance(7)
        _, meta = candidate.action(7)
        assert meta["index"] == 7
        elapsed = []
        for _ in range(20):
            start = time.monotonic()
            env.step(np.array([0, 0, 0, 0, 0, 0, -1]))
            elapsed.append(time.monotonic()-start)
        report = {"renderer": renderer, "original_empty_prefix_bitwise_equal": True,
                  "sampler": projected.sampler, "env_step_seconds": elapsed,
                  "env_mean_s": float(np.mean(elapsed)), "native_init": 0, "formal_states_used": False,
                  "budget_remaining": Budget(root/"runs/latency_v2_budget.json").remaining()}
        publish(directory/"validation.json", report)
        print(json.dumps(report), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
