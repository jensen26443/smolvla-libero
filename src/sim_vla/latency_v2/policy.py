"""Project-local clean-prefix flow projection; upstream sampler stays intact."""
from __future__ import annotations

import time

import numpy as np

from ..latency_eval.policy_adapter import SmolVLAPolicyAdapter
from .schema import InferenceRequest, InferenceResult


def integrate_projected(noise, target, length, steps, velocity):
    """Project BEFORE the velocity field and AFTER every native Euler step."""
    import torch

    x = noise.clone()
    errors = []
    dt = -1.0 / steps
    for step in range(steps):
        x[:, :length] = target[:, :length]
        errors.append((x[:, :length] - target[:, :length]).abs().amax())
        t = torch.tensor(1.0 + step * dt, device=x.device, dtype=torch.float32).expand(x.shape[0])
        x = x + dt * velocity(x, t)
        x[:, :length] = target[:, :length]
        errors.append((x[:, :length] - target[:, :length]).abs().amax())
    return x, torch.stack(errors)


class PrefixPolicyAdapter(SmolVLAPolicyAdapter):
    def __init__(self, **kwargs) -> None:
        from ..assets import libero_assets_dir, project_root
        from ..runtime import import_libero

        import_libero(root=project_root(), assets=libero_assets_dir())
        super().__init__(**kwargs)
        if self.config.num_steps != 10 or self.config.n_obs_steps != 1 or self.config.adapt_to_pi_aloha:
            raise ValueError("audited E1 requires 10 flow steps, one observation, native action representation")
        from lerobot.processor.normalize_processor import NormalizerProcessorStep

        self.action_normalizer = next(step for step in self.preprocessor.steps
                                      if isinstance(step, NormalizerProcessorStep))
        self.last_sampler: dict = {}

    def _preprocess(self, packet):
        # Native torch.from_numpy may share CPU storage. Keep readonly snapshots
        # immutable while giving upstream processors their own writable inputs.
        from dataclasses import replace

        def writable(value):
            if isinstance(value, np.ndarray):
                return value.copy()
            if isinstance(value, dict):
                return {key: writable(item) for key, item in value.items()}
            return value

        return super()._preprocess(replace(packet, images=writable(packet.images),
                                          robot_state=writable(packet.robot_state)))

    def normalize_prefix(self, actions: np.ndarray):
        import torch
        from lerobot.types import TransitionKey

        tensor = torch.as_tensor(np.array(actions, copy=True), device=self.device,
                                 dtype=torch.float32).unsqueeze(0)
        norm = self.action_normalizer({TransitionKey.ACTION: tensor})[TransitionKey.ACTION]
        padded = torch.zeros((1, self.chunk_size, self.max_action_dim), device=self.device)
        padded[:, :len(actions), :7] = norm
        return padded

    def _projected_chunk(self, batch, noise, prefix):
        import torch
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks, populate_queues
        from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        policy, model = self.policy, self.policy.model
        policy.eval()
        batch = policy._prepare_batch(batch)
        policy._queues = populate_queues(policy._queues, batch, exclude_keys=[ACTION])
        for key in batch:
            if key in policy._queues and key != ACTION:
                batch[key] = torch.stack(list(policy._queues[key]), dim=1)
        images, image_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        embeddings, padding, attention = model.embed_prefix(
            images, image_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK], state=state)
        _, cache = model.vlm_with_expert.forward(
            attention_mask=make_att_2d_masks(padding, attention),
            position_ids=torch.cumsum(padding, dim=1) - 1, past_key_values=None,
            inputs_embeds=[embeddings, None], use_cache=model.config.use_cache, fill_kv_cache=True)
        target = self.normalize_prefix(prefix.actions)
        actions, errors = integrate_projected(
            noise, target, len(prefix.actions), model.config.num_steps,
            lambda x, t: model.denoise_step(x_t=x, timestep=t, prefix_pad_masks=padding, past_key_values=cache))
        self.last_sampler = {"sampler": "projected_clean_prefix", "steps": model.config.num_steps,
                             "dt": -1 / model.config.num_steps, "prefix_length": len(prefix.actions),
                             "prefix_sha256": prefix.digest, "valid_dimensions": 7, "padded_dimensions": 25,
                             "projection_errors_before_after": errors.detach().cpu().tolist()}
        return actions[:, :, :7]

    def infer_request(self, request: InferenceRequest) -> InferenceResult:
        import torch

        packet = request.snapshot.packet()
        self.set_episode_seed(request.seed)
        if not len(request.prefix.actions):
            actions = super().infer(packet, request.request_id, request.request_tick)
            self.last_sampler = {"sampler": "original", "steps": 10, "prefix_length": 0,
                                 "prefix_sha256": request.prefix.digest}
        else:
            t0 = time.perf_counter()
            batch = self._preprocess(packet)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            with torch.inference_mode():
                chunk = self._projected_chunk(batch, self._make_noise(request.request_id), request.prefix)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            actions = self.action_spec.clip(self._postprocess_chunk(chunk))
            t3 = time.perf_counter()
            self.last_stage_seconds = {"preprocess_s": t1-t0, "forward_s": t2-t1, "postprocess_s": t3-t2}
            error = float(np.max(np.abs(actions[:len(request.prefix.actions)] - request.prefix.actions)))
            self.last_sampler["environment_prefix_roundtrip_max_error"] = error
            if error > 2e-5:
                raise ValueError(f"native normalization roundtrip error {error}")
        return InferenceResult(request, actions, time.monotonic(), dict(self.last_stage_seconds),
                               dict(self.last_sampler))
