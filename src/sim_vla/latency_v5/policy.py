"""The same sampler; complete render/prepare/inference cost stays in endpoint."""
import time
from dataclasses import replace

from ..latency_v2.schema import ObservationSnapshot
from ..latency_v4.policy import SamplingPolicyAdapter
from .env import SceneRenderer


class RenderPolicyAdapter(SamplingPolicyAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.renderer = SceneRenderer()

    def infer_request(self, request):
        entered_ns = time.monotonic_ns()
        request.snapshot.validate()
        observation = self.renderer.render(request.snapshot)
        render_completed_ns = time.monotonic_ns()
        # The controller-owned robot state is retained exactly; the replica
        # supplies only cameras at the frozen state, never evaluator telemetry.
        observation["robot_state"] = request.snapshot.robot_state
        pixels = ObservationSnapshot.capture(request.snapshot.episode, request.snapshot.tick, observation,
                                             request.snapshot.language)
        rendered_request = replace(request, snapshot=pixels)
        result = super().infer_request(rendered_request)
        completed_ns = time.monotonic_ns()
        result.request = request
        result.ready_wall = time.monotonic()
        result.worker_completed_monotonic_ns = completed_ns
        result.stages.update(render_scene_s=(render_completed_ns - entered_ns) / 1e9,
                             worker_queue_and_ipc_s=max(0., (entered_ns - request.submit_monotonic_ns) / 1e9),
                             policy_endpoint_s=(completed_ns - entered_ns) / 1e9)
        result.sampler.update(worker_entered_monotonic_ns=entered_ns,
                              rendered_observation_sha256=pixels.digest,
                              rendered_scene_sha256=request.snapshot.scene_sha256,
                              rendered_visual_sha256=request.snapshot.visual_sha256,
                              input_physical_tick=request.snapshot.tick,
                              camera_sampling=observation["sampling"],
                              observation_protocol="frozen_full_scene_rendered_in_worker")
        result.rendered_images = {key: value.copy() for key, value in pixels.images.items()}
        completed_ns = time.monotonic_ns()
        result.worker_completed_monotonic_ns = completed_ns
        result.ready_wall = time.monotonic()
        result.stages["policy_endpoint_s"] = (completed_ns - entered_ns) / 1e9
        return result
