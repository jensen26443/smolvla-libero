"""Frozen SmolVLA inference adapter: snapshot packet -> (H, dim) action chunk.

The adapter reproduces the official eval preprocessing chain verbatim
(``preprocess_observation`` -> task injection equivalent to ``add_envs_task``
-> ``LiberoProcessorStep`` -> policy preprocessor), then calls
``predict_action_chunk`` directly — the public API the official
``select_action`` itself uses — so the full 50-step chunk is available to
chunk-executing methods instead of being truncated to one step. Flow-matching
noise is drawn from a per-request seeded generator, making every trial
reproducible. Per-stage GPU timings (preprocess / forward / postprocess) are
recorded with proper synchronization.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import ActionSpec, ObservationPacket


class SmolVLAPolicyAdapter:
    checkpoint_label: str

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        device: str = "cuda",
        action_spec: ActionSpec | None = None,
        warmup_calls: int = 3,
        seed: int = 0,
        task_ids_for_cfg: tuple[int, ...] = (1, 8),
    ) -> None:
        self.checkpoint = Path(checkpoint).resolve()
        if not self.checkpoint.is_dir():
            raise FileNotFoundError(f"checkpoint not found: {self.checkpoint}")
        self.checkpoint_label = str(self.checkpoint)
        self.device = device
        self.action_spec = action_spec or ActionSpec()
        self._seed = int(seed)
        self._warmup_counter = 0
        self.last_stage_seconds: dict[str, float] = {}

        # Heavy imports stay inside __init__ so merely importing this module
        # (e.g. in unit tests) never requires the simulation stack.
        import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.policies.factory import make_policy, make_pre_post_processors

        env_cfg = LiberoEnvConfig(
            task="libero_goal",
            task_ids=list(task_ids_for_cfg),
            control_mode="relative",
            episode_length=300,
            max_parallel_tasks=1,
            observation_width=256,
            observation_height=256,
        )
        cfg = PreTrainedConfig.from_pretrained(self.checkpoint)
        cfg.pretrained_path = self.checkpoint
        cfg.device = device
        if hasattr(cfg, "use_amp"):
            cfg.use_amp = False
        self.config = cfg
        self.policy = make_policy(cfg=cfg, env_cfg=env_cfg)
        preprocessor_overrides = {
            "device_processor": {"device": str(self.policy.config.device)},
            "rename_observations_processor": {"rename_map": {}},
        }
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=self.checkpoint,
            preprocessor_overrides=preprocessor_overrides,
        )
        self.env_preprocessor, self.env_postprocessor = make_env_pre_post_processors(
            env_cfg=env_cfg, policy_cfg=cfg
        )
        self.chunk_size = int(self.policy.config.chunk_size)
        self.max_action_dim = int(self.policy.config.max_action_dim)
        if self.chunk_size != self.action_spec.horizon:
            raise ValueError(
                f"checkpoint chunk_size {self.chunk_size} != action spec horizon {self.action_spec.horizon}"
            )
        self._warmed = False
        self._warmup_calls = int(warmup_calls)
        self._torch = None

    # -- official-path plumbing -------------------------------------------------

    def _preprocess(self, packet: ObservationPacket) -> dict[str, Any]:
        from lerobot.envs.utils import preprocess_observation

        # The official eval runs through a SyncVectorEnv, so every robot_state
        # array arrives batched; our single-env packets need the leading batch
        # dim restored before LiberoProcessorStep (it asserts (B, 4) quaternions).
        robot_state: dict[str, Any] | None = None
        if packet.robot_state is not None:
            robot_state = {
                group: {
                    key: value[None, ...] if isinstance(value, np.ndarray) and value.ndim == 1 else value
                    for key, value in fields.items()
                }
                for group, fields in packet.robot_state.items()
                if isinstance(fields, dict)
            }
        raw: dict[str, Any] = {"pixels": packet.images}
        if robot_state is not None:
            raw["robot_state"] = robot_state
        observation = preprocess_observation(raw)
        # add_envs_task equivalent for a single environment.
        observation["task"] = [packet.instruction]
        observation = self.env_preprocessor(observation)
        return self.preprocessor(observation)

    def _postprocess_chunk(self, chunk: Any) -> np.ndarray:
        """De-normalize a (1, H, dim) chunk exactly as the official executor
        would de-normalize its (1, dim) per-step action."""

        processed = self.postprocessor(chunk)
        if isinstance(processed, dict):
            from lerobot.constants import ACTION

            processed = processed[ACTION]
        array = processed.detach().to("cpu").numpy()
        if array.ndim == 2:
            array = array[None, ...]
        if array.shape != (1, self.chunk_size, self.action_spec.dim):
            raise ValueError(
                f"postprocessed chunk has shape {array.shape}, expected "
                f"(1, {self.chunk_size}, {self.action_spec.dim}); the checkpoint postprocessor "
                "does not broadcast over the chunk horizon and needs a per-step fallback"
            )
        return array[0].astype(np.float64)

    def _make_noise(self, request_id: int) -> Any:
        import torch

        generator_seed = (self._seed * 1_000_003 + request_id) % (2**31 - 1)
        generator = torch.Generator(device=self.policy.config.device)
        generator.manual_seed(int(generator_seed))
        return torch.randn(
            (1, self.chunk_size, self.max_action_dim),
            dtype=torch.float32,
            device=self.policy.config.device,
            generator=generator,
        )

    # -- adapter protocol --------------------------------------------------------

    def reset(self) -> None:
        self.policy.reset()

    def set_episode_seed(self, seed: int) -> None:
        """Noise streams are request-indexed; the seed parametrizes the stream."""

        self._seed = int(seed)

    def warmup(self, packet: ObservationPacket | None = None) -> dict[str, float]:
        """Run a few inferences before timed episodes (never on live states).

        Uses the provided packet or a zero-filled stand-in; warmup happens
        before episode timing starts and cannot observe future states.
        """

        if self._warmed:
            return {}
        import torch

        if packet is None:
            images = {
                "image": np.zeros((256, 256, 3), dtype=np.uint8),
                "image2": np.zeros((256, 256, 3), dtype=np.uint8),
            }
            robot_state = {
                "eef": {"pos": np.zeros(3), "quat": np.array([1.0, 0.0, 0.0, 0.0])},
                "gripper": {"qpos": np.zeros(2)},
            }
            packet = ObservationPacket(
                packet_id=-1,
                capture_tick=-1,
                arrive_tick=-1,
                images=images,
                robot_state=robot_state,
                instruction="warmup",
                modality_capture_ticks={},
            )
        timings: list[float] = []
        with torch.inference_mode():
            for _ in range(self._warmup_calls):
                started = time.perf_counter()
                self.infer(packet, request_id=-1, request_tick=-1)
                if self.policy.config.device.startswith("cuda"):
                    torch.cuda.synchronize()
                timings.append(time.perf_counter() - started)
        self._warmed = True
        return {"warmup_call_seconds": timings[-1] if timings else None}

    def infer(self, packet: ObservationPacket, request_id: int, request_tick: int) -> np.ndarray:
        import torch

        cuda = self.policy.config.device.startswith("cuda")
        if request_id < 0:  # warmup: never collides with the live request stream
            request_id = self._warmup_counter
            self._warmup_counter += 1

        t0 = time.perf_counter()
        model_input = self._preprocess(packet)
        if cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        noise = self._make_noise(request_id)
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(model_input, noise=noise)
        if cuda:
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        actions = self._postprocess_chunk(chunk)
        t3 = time.perf_counter()
        self.last_stage_seconds = {
            "preprocess_s": t1 - t0,
            "forward_s": t2 - t1,
            "postprocess_s": t3 - t2,
        }
        return self.action_spec.clip(actions)
