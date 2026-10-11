"""One development reset, fixed observation/noise, no scored rollout or tuning."""
import json
from pathlib import Path

import numpy as np

from sim_vla.assets import project_root
from sim_vla.latency_v2 import policy as policy_module
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
    directory = root/"runs/latency_v2_conditioning_probe"
    if directory.exists():
        raise FileExistsError("preserve the existing development conditioning probe")
    attempt = Budget(root/"runs/latency_v2_budget.json").claim("S3:conditioning:dev_init0", "S3_validation")
    publish(directory/"attempt.json", {"attempt_number": attempt, "development_init": 0, "seed": 42,
                                      "policy_rollout_scored": False})
    policy = PrefixPolicyAdapter(checkpoint=root/CHECKPOINT, warmup_calls=3)
    policy.warmup()
    env = ControlledEnv(task_key="goal_bowl_plate")
    try:
        observation = env.start_episode(0, 42)
        snapshot = ObservationSnapshot.capture("S3_development_probe", 0, observation, env.instruction)
        empty = CommittedPrefix(0, np.zeros((0, 7)), (), digest_arrays(np.zeros((0, 7))))
        request = InferenceRequest(attempt, 0, snapshot, 0, empty, 42, 0.)
        policy.reset()
        native = policy.infer_request(request).actions

        def unprojected(noise, target, length, steps, velocity):
            assert length == 0
            x = noise.clone()
            dt = -1./steps
            for step in range(steps):
                time = torch.tensor(1.+step*dt, dtype=torch.float32, device=x.device).expand(x.shape[0])
                x = x+dt*velocity(x, time)
            # Empty prefix has no projection positions; this is a wrapper-equivalence check.
            return x, torch.zeros(2*steps, device=x.device)

        policy.reset()
        policy.set_episode_seed(42)
        original = policy_module.integrate_projected
        try:
            policy_module.integrate_projected = unprojected
            with torch.inference_mode():
                chunk = policy._projected_chunk(policy._preprocess(snapshot.packet()), policy._make_noise(0), empty)
            private = policy.action_spec.clip(policy._postprocess_chunk(chunk))
        finally:
            policy_module.integrate_projected = original  # Only this isolated process is instrumented.

        prefixes = [native[:7].copy(), native[:7].copy()]
        prefixes[1][:, :3] *= -1
        latent, environmental, samplers = [], [], []
        for actions in prefixes:
            prefix = CommittedPrefix(0, actions, (), digest_arrays(actions))
            policy.reset()
            policy.set_episode_seed(42)
            with torch.inference_mode():
                chunk = policy._projected_chunk(policy._preprocess(snapshot.packet()), policy._make_noise(0), prefix)
            latent.append(chunk.detach().cpu().numpy().copy())
            output = policy.action_spec.clip(policy._postprocess_chunk(chunk))
            environmental.append(output)
            sampler = dict(policy.last_sampler)
            sampler["environment_prefix_roundtrip_max_error"] = float(np.max(np.abs(output[:7]-actions)))
            samplers.append(sampler)
        latent_difference = float(np.max(np.abs(latent[0][:, 7:]-latent[1][:, 7:])))
        native_equal = bool(np.array_equal(native, private))
        report = {"stage": "S3", "attempt_number": attempt, "development_init_index": 0, "seed": 42,
            "scored_rollout": False, "formal_states_accessed": False, "environment_resets": 1,
            "fixed_observation_sha256": snapshot.digest, "fixed_request_noise_ordinal": 0,
            "checkpoint_sha256": file_sha256(root/CHECKPOINT/"model.safetensors"),
            "source_policy_sha256": file_sha256(Path(policy_module.__file__)),
            "probe_script_sha256": file_sha256(Path(__file__)),
            "private_wrapper_without_projection_bitwise_native_equal": native_equal,
            "private_wrapper_max_environment_difference": float(np.max(np.abs(native-private))),
            "prefix_intervention": "same native first 7 actions, flip translation signs only; fixed images/state/language/noise",
            "normalized_suffix_linf_difference": latent_difference,
            "environment_suffix_linf_difference": float(np.max(np.abs(environmental[0][7:]-environmental[1][7:]))),
            "prefix_coupling_detected": latent_difference > 1e-7, "samplers": samplers,
            "architecture": {"attention_mode": policy.config.attention_mode,
                             "self_attn_every_n_layers": policy.config.self_attn_every_n_layers},
            "scope": "interface and causal input-effect probe, not task effectiveness or a parameter selection"}
        np.savez(directory/"arrays.npz", native=native, private_unprojected=private,
                 prefix_a=prefixes[0], prefix_b=prefixes[1], latent_a=latent[0], latent_b=latent[1],
                 environment_a=environmental[0], environment_b=environmental[1])
        publish(directory/"probe.json", report)
        if not native_equal:
            raise AssertionError("private sampler wrapper differs from the native path before projection")
        if any(max(row["projection_errors_before_after"]) > 0
               or row["environment_prefix_roundtrip_max_error"] > 2e-5 for row in samplers):
            raise AssertionError("native normalization or per-step prefix projection failed")
        publish(directory/"complete.json", {"native_equivalence_passed": True,
                                           "prefix_coupling_detected": report["prefix_coupling_detected"]})
        print(json.dumps({key: value for key, value in report.items() if key != "samplers"}, ensure_ascii=False))
    finally:
        env.close()


if __name__ == "__main__":
    main()
