"""Explicit step counts with unchanged upstream preprocessing and RTC formulas."""
import time

import numpy as np

from ..latency_eval.policy_adapter import SmolVLAPolicyAdapter
from ..latency_v2.policy import PrefixPolicyAdapter
from ..latency_v2.schema import InferenceResult
from .schema import SAMPLING_STEPS


class SamplingPolicyAdapter(PrefixPolicyAdapter):
    def __init__(self, *, sampling_steps=10, **kwargs):
        import torch

        torch.set_num_threads(4)
        super().__init__(**kwargs)
        self.bound_steps = sampling_steps
        if sampling_steps is not None and sampling_steps not in SAMPLING_STEPS:
            raise ValueError("unsupported sampling steps")
        for parameter in self.policy.parameters():
            parameter.requires_grad_(False)
        self.set_sampling_steps(sampling_steps or 10)

    def set_sampling_steps(self, steps):
        if steps not in SAMPLING_STEPS or (self.bound_steps is not None and steps != self.bound_steps):
            raise ValueError("sampling steps differ from the worker configuration")
        for cfg in (self.config, self.policy.config, self.policy.model.config):
            cfg.num_steps = steps
        if any(cfg.num_steps != steps for cfg in (self.config, self.policy.config, self.policy.model.config)):
            raise RuntimeError("sampling configuration did not reach the model")

    def _disable_rtc(self):
        self.policy.config.rtc_config = None
        self.policy.init_rtc_processor()

    def infer_request(self, request):
        import torch

        entered_ns = time.monotonic_ns()
        self.set_sampling_steps(request.sampling_steps)
        self.set_episode_seed(request.seed)
        prefix_length = len(request.prefix.actions)
        self._disable_rtc()
        if request.sampler_mode.startswith("RTC_") and prefix_length:
            from lerobot.configs.types import RTCAttentionSchedule
            from lerobot.policies.rtc.configuration_rtc import RTCConfig

            if request.guide_actions is None or len(request.guide_actions) < prefix_length:
                raise ValueError("RTC requires actual old actions throughout the commitment")
            if torch.is_inference_mode_enabled():
                raise RuntimeError("RTC local autograd cannot run in inference_mode")
            guide_length = len(request.guide_actions)
            self.policy.config.rtc_config = RTCConfig(enabled=True, max_guidance_weight=10.,
                prefix_attention_schedule=RTCAttentionSchedule.EXP, execution_horizon=guide_length)
            self.policy.init_rtc_processor()
            t0 = time.perf_counter()
            batch = self._preprocess(request.snapshot.packet())
            target = self.normalize_prefix(request.guide_actions)[:, :guide_length]
            self._synchronize()
            t1 = time.perf_counter()
            with torch.no_grad():
                chunk = self.policy.predict_action_chunk(batch, noise=self._make_noise(request.request_id),
                    inference_delay=prefix_length, prev_chunk_left_over=target, execution_horizon=guide_length)
            self._synchronize()
            t2 = time.perf_counter()
            actions = self.action_spec.clip(self._postprocess_chunk(chunk))
            t3 = time.perf_counter()
            stages = {"preprocess_s": t1-t0, "forward_s": t2-t1, "postprocess_s": t3-t2}
            sampler = {"sampler": "lerobot_0.5.1_native_rtc_guidance", "guide_length": guide_length,
                "guidance_weight": 10., "schedule": "EXP", "soft_overlap": request.soft_overlap,
                "environment_generated_prefix_max_error": float(np.max(
                    np.abs(actions[:prefix_length]-request.prefix.actions))), "padding_guided": True}
        elif request.sampler_mode == "HARD" and prefix_length:
            result = super().infer_request(request)
            actions, stages, sampler = result.actions, result.stages, result.sampler
        else:
            actions = SmolVLAPolicyAdapter.infer(self, request.snapshot.packet(), request.request_id,
                                               request.request_tick)
            stages = dict(self.last_stage_seconds)
            sampler = {"sampler": "original", "guide_length": 0}
        gradients = any(parameter.grad is not None for parameter in self.policy.parameters())
        if gradients:
            raise RuntimeError("RTC accumulated parameter gradients")
        completed_ns = time.monotonic_ns()
        stages.update(worker_queue_and_ipc_s=max(0, (entered_ns-request.submit_monotonic_ns)/1e9),
                      policy_endpoint_s=(completed_ns-entered_ns)/1e9)
        sampler.update(method=request.sampler_mode, steps=request.sampling_steps, dt=-1/request.sampling_steps,
            prefix_length=prefix_length, prefix_sha256=request.prefix.digest,
            parameter_gradients_present=gradients, worker_bound_steps=self.bound_steps,
            worker_entered_monotonic_ns=entered_ns,
            gpu_allocated_peak_bytes=torch.cuda.max_memory_allocated() if self.device.startswith("cuda") else None,
            gpu_reserved_peak_bytes=torch.cuda.max_memory_reserved() if self.device.startswith("cuda") else None)
        return InferenceResult(request, actions, time.monotonic(), stages, sampler,
                               worker_completed_monotonic_ns=completed_ns)

    def _synchronize(self):
        if self.device.startswith("cuda"):
            import torch

            torch.cuda.synchronize()
