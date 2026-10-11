"""One-batch SmolVLA forward/backward and save/load compatibility check."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from .assets import model_snapshot_dir, project_root
from .protocol import SEED, dump_json
from .runtime import import_libero


def _clone_batch(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clone_batch(item) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return value.clone()
    return value


def _tensor_shape(value: Any) -> list[int] | None:
    return list(value.shape) if isinstance(value, torch.Tensor) else None


def _as_action_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, dict):
        value = value["action"]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"processor returned {type(value).__name__}, expected an action tensor")
    return value


def run_model_smoke(
    *,
    model_path: str | Path | None = None,
    output: str | Path = "runs/model_smoke.json",
    task_id: int = 8,
    seed: int = SEED,
    device: str = "cuda",
    save_load: bool = True,
) -> dict[str, Any]:
    """Run inference, one optimizer step, and optional local save/load."""

    started = time.time()
    selected_model = Path(model_path) if model_path is not None else model_snapshot_dir()
    if not selected_model.is_dir():
        raise FileNotFoundError(
            f"model snapshot not found at {selected_model}; run `sim-vla prepare-assets --kind model` first"
        )
    import_libero(root=project_root())

    # Heavy imports are delayed until the project-local LIBERO config exists.
    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401  (register policy choice)
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.envs.utils import add_envs_task, preprocess_observation
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    selected_model = selected_model.resolve()
    env_cfg = LiberoEnv(
        task="libero_goal",
        task_ids=[task_id],
        observation_width=256,
        observation_height=256,
        control_mode="relative",
    )
    cfg = PreTrainedConfig.from_pretrained(selected_model)
    cfg.pretrained_path = selected_model
    cfg.device = device
    cfg.push_to_hub = False
    policy = make_policy(cfg, env_cfg=env_cfg)
    trainable = sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in policy.parameters())

    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    vec = envs["libero_goal"][task_id]
    env_preprocessor, _env_postprocessor = make_env_pre_post_processors(env_cfg, policy.config)
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=selected_model)
    try:
        observation, _ = vec.reset(seed=[seed])
        observation = preprocess_observation(observation)
        observation = add_envs_task(vec, observation)
        observation = env_preprocessor(observation)
        model_input = preprocessor(observation)
        inference_input = {
            key: _clone_batch(value) for key, value in model_input.items() if key not in {"action", "action_is_pad"}
        }
        policy.reset()
        with torch.inference_mode():
            raw_action = policy.select_action(_clone_batch(inference_input))
        action = _as_action_tensor(postprocessor(raw_action))
        if tuple(action.shape) != (1, 7):
            raise AssertionError(f"postprocessor returned action shape {tuple(action.shape)}, expected (1, 7)")
        _, reward, terminated, truncated, info = vec.step(action.detach().cpu().numpy())

        train_batch = _clone_batch(inference_input)
        train_batch["action"] = torch.zeros(
            (1, policy.config.chunk_size, 7), dtype=torch.float32, device=policy.config.device
        )
        train_batch["action_is_pad"] = torch.zeros(
            (1, policy.config.chunk_size), dtype=torch.bool, device=policy.config.device
        )
        policy.train()
        optimizer = torch.optim.AdamW(
            (parameter for parameter in policy.parameters() if parameter.requires_grad),
            lr=policy.config.optimizer_lr,
        )
        optimizer.zero_grad(set_to_none=True)
        loss, loss_details = policy(train_batch)
        loss.backward()
        gradients = [parameter.grad.detach().norm() for parameter in policy.parameters() if parameter.grad is not None]
        gradient_norm = float(torch.stack(gradients).norm().item()) if gradients else 0.0
        optimizer.step()
        policy.eval()

        result: dict[str, Any] = {
            "status": "passed",
            "protocol_version": "sim-vla-libero-v1",
            "model_path": str(selected_model),
            "task_id": task_id,
            "device": str(policy.config.device),
            "total_parameters": total,
            "trainable_parameters": trainable,
            "input_shapes": {key: _tensor_shape(value) for key, value in model_input.items()},
            "raw_action_shape": list(raw_action.shape),
            "action": action.detach().cpu().tolist(),
            "step": {
                "reward": float(reward[0]),
                "terminated": bool(terminated[0]),
                "truncated": bool(truncated[0]),
                "is_success": bool(info.get("is_success", [False])[0]),
            },
            "train": {
                "loss": float(loss.detach().cpu().item()),
                "loss_details": loss_details,
                "gradient_norm": gradient_norm,
                "optimizer_step": 1,
            },
            "save_load": {"requested": save_load, "verified": False},
            "elapsed_seconds": time.time() - started,
        }

        if save_load:
            save_dir = project_root() / ".cache" / "model-smoke-checkpoint"
            if save_dir.exists():
                shutil.rmtree(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(save_dir)
            preprocessor.save_pretrained(save_dir, push_to_hub=False, config_filename="policy_preprocessor.json")
            postprocessor.save_pretrained(save_dir, push_to_hub=False, config_filename="policy_postprocessor.json")
            reference_batch = _clone_batch(inference_input)
            zero_noise = torch.zeros(
                (1, policy.config.chunk_size, policy.config.max_action_dim),
                dtype=torch.float32,
                device=policy.config.device,
            )
            policy.reset()
            with torch.inference_mode():
                reference = policy.predict_action_chunk(reference_batch, noise=zero_noise).detach().cpu()
            del policy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            loaded_cfg = PreTrainedConfig.from_pretrained(save_dir)
            loaded_cfg.pretrained_path = save_dir
            loaded_cfg.device = device
            loaded = make_policy(loaded_cfg, env_cfg=env_cfg)
            loaded.reset()
            with torch.inference_mode():
                reloaded = loaded.predict_action_chunk(_clone_batch(inference_input), noise=zero_noise).detach().cpu()
            max_abs_diff = float((reference - reloaded).abs().max().item())
            result["save_load"] = {
                "requested": True,
                "verified": True,
                "preprocessor": (save_dir / "policy_preprocessor.json").is_file(),
                "postprocessor": (save_dir / "policy_postprocessor.json").is_file(),
                "max_abs_action_chunk_diff": max_abs_diff,
                "checkpoint_dir": str(save_dir),
            }
            del loaded
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        vec.close()

    dump_json(result, output)
    return result


def model_smoke_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run SmolVLA inference/train/save-load smoke")
    parser.add_argument("--model-path")
    parser.add_argument("--output", default="runs/model_smoke.json")
    parser.add_argument("--task-id", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-save-load", action="store_true")
    args = parser.parse_args(argv)
    result = run_model_smoke(
        model_path=args.model_path,
        output=args.output,
        task_id=args.task_id,
        seed=args.seed,
        device=args.device,
        save_load=not args.no_save_load,
    )
    print(f"SmolVLA model smoke passed; record written to {args.output}; loss={result['train']['loss']:.6f}")
    return 0
