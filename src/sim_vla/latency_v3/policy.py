"""Native LeRobot RTC guidance and the unchanged PLAN-04 projection sampler."""
import time

import numpy as np

from ..latency_v2.policy import PrefixPolicyAdapter
from ..latency_v2.schema import InferenceResult


class ResearchPolicyAdapter(PrefixPolicyAdapter):
    def __init__(self, **kwargs):
        import torch

        torch.set_num_threads(4)
        super().__init__(**kwargs)
        for parameter in self.policy.parameters():
            parameter.requires_grad_(False)
        self.probe_debug = False

    def disable_rtc(self):
        self.policy.config.rtc_config = None
        self.policy.init_rtc_processor()

    def infer_request(self, request):
        import torch

        if request.sampler_mode != "RTC_GUIDED" or not len(request.prefix.actions):
            self.disable_rtc()
            result = super().infer_request(request)
            result.sampler["method"] = request.sampler_mode
            return result

        from lerobot.configs.types import RTCAttentionSchedule
        from lerobot.policies.rtc.configuration_rtc import RTCConfig

        if request.guide_actions is None or len(request.guide_actions) < len(request.prefix.actions):
            raise ValueError("RTC requires real old actions throughout its commitment")
        if torch.is_inference_mode_enabled():
            raise RuntimeError("native RTC autograd must not run inside inference_mode")
        length = len(request.prefix.actions)
        guide_length = len(request.guide_actions)
        self.policy.config.rtc_config = RTCConfig(enabled=True, max_guidance_weight=10.,
            prefix_attention_schedule=RTCAttentionSchedule.EXP, execution_horizon=guide_length,
            debug=self.probe_debug)
        self.policy.init_rtc_processor()
        self.set_episode_seed(request.seed)
        t0 = time.perf_counter()
        batch = self._preprocess(request.snapshot.packet())
        target = self.normalize_prefix(request.guide_actions)[:, :guide_length]
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        # no_grad allows the official denoise_step's local enable_grad context.
        with torch.no_grad():
            chunk = self.policy.predict_action_chunk(batch, noise=self._make_noise(request.request_id),
                inference_delay=length, prev_chunk_left_over=target, execution_horizon=guide_length)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        actions = self.action_spec.clip(self._postprocess_chunk(chunk))
        sampler = {"sampler": "lerobot_0.5.1_native_rtc_guidance", "method": request.sampler_mode,
            "steps": 10, "dt": -.1, "prefix_length": length, "guide_length": guide_length,
            "guidance_weight": 10., "schedule": "EXP", "soft_overlap": request.soft_overlap,
            "environment_generated_prefix_max_error": float(np.max(
                np.abs(actions[:length]-request.prefix.actions))),
            "parameter_gradients_present": any(p.grad is not None for p in self.policy.parameters()),
            "padding_guided": True}
        if self.probe_debug:
            sampler["guidance_steps"] = [step.to_dict() for step in self.policy.rtc_processor.get_all_debug_steps()]
        t3 = time.perf_counter()
        stages = {"preprocess_s": t1-t0, "forward_s": t2-t1, "postprocess_s": t3-t2}
        return InferenceResult(request, actions, time.monotonic(), stages, sampler)
