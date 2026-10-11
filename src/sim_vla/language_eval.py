"""Language-conditioned LIBERO evaluation with frozen semantic paraphrases.

The official lerobot-eval entry point always takes task language from the
environment. E5 therefore uses the same LeRobot rollout implementation but
replaces only task_description on each selected environment. The task,
initial-state sequence, action interface, and official success predicate
remain unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .assets import project_root
from .protocol import SEED, dump_json, load_json
from .runtime import import_libero


def load_paraphrase_manifest(path: str | Path) -> dict[int, dict[str, Any]]:
    """Load and validate the frozen task-id to paraphrase mapping."""

    payload = load_json(path)
    if not isinstance(payload, dict) or payload.get("condition") != "E5":
        raise ValueError(f"not an E5 paraphrase manifest: {path}")
    rows = payload.get("instructions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("E5 paraphrase manifest must contain a non-empty instructions list")

    by_task_id: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("E5 instruction rows must be objects")
        task_id = row.get("suite_task_id")
        original = row.get("original")
        paraphrase = row.get("paraphrase")
        if not isinstance(task_id, int) or task_id < 0:
            raise ValueError(f"invalid suite_task_id in E5 manifest: {task_id!r}")
        if not isinstance(original, str) or not original.strip():
            raise ValueError(f"missing original instruction for task {task_id}")
        if not isinstance(paraphrase, str) or not paraphrase.strip():
            raise ValueError(f"missing paraphrase for task {task_id}")
        if original.strip() == paraphrase.strip():
            raise ValueError(f"E5 paraphrase is identical to original for task {task_id}")
        if task_id in by_task_id:
            raise ValueError(f"duplicate E5 suite_task_id: {task_id}")
        by_task_id[task_id] = row
    return by_task_id


def run_paraphrase_eval(
    *,
    checkpoint: str | Path,
    paraphrase_manifest: str | Path = "manifests/semantic_paraphrases.json",
    output_dir: str | Path = "results/e5_formal",
    episodes: int = 30,
    seed: int = SEED,
    device: str = "cuda",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run E5 with the official LeRobot rollout and frozen paraphrases."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    selected_checkpoint = Path(checkpoint).resolve()
    if not selected_checkpoint.is_dir():
        raise FileNotFoundError(f"E5 checkpoint does not exist: {selected_checkpoint}")
    output = Path(output_dir)
    eval_info_path = output / "eval_info.json"
    if eval_info_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing E5 result: {eval_info_path}")

    by_task_id = load_paraphrase_manifest(paraphrase_manifest)
    import_libero(root=project_root())

    # Delay heavyweight imports so manifest/protocol tests do not need MuJoCo.
    # Register SmolVLA's policy choice before loading the checkpoint config.
    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.envs.utils import close_envs
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_eval import eval_policy_all

    task_ids = sorted(by_task_id)
    env_cfg = LiberoEnv(
        task="libero_goal",
        task_ids=task_ids,
        control_mode="relative",
        max_parallel_tasks=1,
    )
    cfg = PreTrainedConfig.from_pretrained(selected_checkpoint)
    cfg.pretrained_path = selected_checkpoint
    cfg.device = device
    if hasattr(cfg, "use_amp"):
        cfg.use_amp = False

    output.mkdir(parents=True, exist_ok=True)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    try:
        # make_env returns one SyncVectorEnv per selected task. The override is
        # persistent across resets because LiberoEnv.reset does not replace it.
        suite_envs = envs["libero_goal"]
        for task_id, vec_env in suite_envs.items():
            if task_id not in by_task_id:
                raise ValueError(f"environment task {task_id} lacks an E5 paraphrase")
            vec_env.envs[0].task_description = by_task_id[task_id]["paraphrase"]

        policy = make_policy(cfg=cfg, env_cfg=env_cfg)
        preprocessor_overrides = {
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": {}},
        }
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=selected_checkpoint,
            preprocessor_overrides=preprocessor_overrides,
        )
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=env_cfg, policy_cfg=cfg
        )
        with torch.no_grad():
            info = eval_policy_all(
                envs=envs,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=episodes,
                max_episodes_rendered=10,
                videos_dir=output / "videos",
                start_seed=seed,
                max_parallel_tasks=1,
            )
    finally:
        close_envs(envs)

    with eval_info_path.open("w", encoding="utf-8") as handle:
        json.dump(info, handle, ensure_ascii=False, indent=2)
    dump_json(
        {
            "condition": "E5",
            "protocol_version": "sim-vla-libero-v1",
            "checkpoint": str(selected_checkpoint),
            "paraphrase_manifest": str(Path(paraphrase_manifest)),
            "task_ids": task_ids,
            "episodes_per_task": episodes,
            "seed": seed,
            "control_mode": "relative",
            "success_predicate": "official LIBERO task predicate, unchanged",
        },
        output / "protocol.json",
    )
    return info


def language_eval_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run E5 semantic-paraphrase LIBERO evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="manifests/semantic_paraphrases.json")
    parser.add_argument("--output-dir", default="results/e5_formal")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    info = run_paraphrase_eval(
        checkpoint=args.checkpoint,
        paraphrase_manifest=args.manifest,
        output_dir=args.output_dir,
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
        overwrite=args.overwrite,
    )
    print(f"E5 evaluation complete; output={args.output_dir}; overall={info['overall']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(language_eval_main())
