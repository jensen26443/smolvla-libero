"""Scalar-only control observations and a separate frozen-scene render owner."""
import os
import time

import numpy as np

from ..assets import libero_assets_dir, project_root
from ..latency_v2.env import ControlledEnv
from ..latency_v2.model_poses import ModelPoseBank, body_names
from ..latency_v2.schema import digest_arrays
from ..runtime import import_libero
from ..tasks import TASK_BY_KEY
from .schema import SceneSnapshot

CAMERAS = {"agentview_image": "image", "robot0_eye_in_hand_image": "image2"}


def local_libero(task_key):
    import_libero(root=project_root(), assets=libero_assets_dir())
    from lerobot.envs.libero import LiberoEnv
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    class SingleResetLibero(LiberoEnv):
        def _make_envs_task(self, task_suite, task_id=0):
            from pathlib import Path

            from libero.libero import get_libero_path

            task = task_suite.get_task(task_id)
            self.task, self.task_description = task.name, task.language
            path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            env = OffScreenRenderEnv(bddl_file_name=str(path), camera_heights=self.observation_height,
                                     camera_widths=self.observation_width)
            # Construction is a charged attempt. No extra unrecorded initial
            # reset, and no wrapper loop hiding placement/reset failures.
            env.reset = env.env.reset
            return env

    task = TASK_BY_KEY[task_key]
    suite = benchmark.get_benchmark_dict()[task.suite]()
    return SingleResetLibero(task_suite=suite, task_id=task.suite_task_id, task_suite_name=task.suite,
                            episode_length=300, camera_name=",".join(CAMERAS), obs_type="pixels_agent_pos",
                            observation_width=256, observation_height=256, init_states=True,
                            episode_index=0, n_envs=1, control_mode="relative")


class StateOnlyEnv(ControlledEnv):
    def __init__(self, *, task_key, profile_env=False):
        from ..latency_v2.env_timing import EnvTiming

        task = TASK_BY_KEY[task_key]
        self.task_key, self.task_id = task_key, task.suite_task_id
        self._env = local_libero(task_key)
        self.instruction = str(self._env.task_description).lower()
        self.episode_length, self.n_resets, self._current_init_index = 300, 0, None
        self.observation_mode = "on_request"
        self._tick = self._sampling_tick = 0
        self._sampling, self.last_sample_timing, self.last_step_timing = {}, {}, {}
        self._sampling_cameras, self._defer_cameras = True, False
        self._rendered_images = None
        self.last_observation, self.model_pose_bank = None, None
        self.timing = EnvTiming() if profile_env else None
        self._attach_sampling()

    def focus(self):
        context = self._env._env.sim._render_context_offscreen
        if context is not None:
            context.gl_ctx.make_current()

    def _attach_sampling(self):
        super()._attach_sampling()
        domain = self._env._env.env
        original = domain._update_observables
        if getattr(original, "_state_only_owner", None) is self:
            return

        def update(force=False):
            if not self._defer_cameras:
                return original(force=force)
            for name, observable in domain._observables.items():
                if name not in CAMERAS:
                    observable.update(timestep=domain.model_timestep, obs_cache=domain._obs_cache, force=force)

        update._latency_camera_owner = self
        update._state_only_owner = self
        domain._update_observables = update

    def start_episode(self, init_index, seed, states=None):
        self._defer_cameras = False
        self.focus()
        observation = super().start_episode(init_index, seed, states=states)
        self._rendered_images = {key: value.copy() for key, value in observation["pixels"].items()}
        self._defer_cameras = True
        return observation

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        # Scalar-only updates retain upstream camera caches from reset. Keep
        # the delivered worker images alongside their own sampling metadata.
        observation["pixels"].update(self._rendered_images)
        self.last_observation = observation
        return observation, reward, terminated, truncated, info

    def freeze_snapshot(self, tick, episode):
        if tick != self._tick:
            raise ValueError("state snapshot tick differs from the physical boundary")
        started_ns = time.monotonic_ns()
        domain = self._env._env.env
        self._defer_cameras = True
        domain.sim.forward()
        domain._update_observables(force=True)
        observation = self._env._format_raw_obs(domain._get_observations(force_update=False))
        sampled = {"state": {"tick": tick, "sim_time": float(domain.sim.data.time),
                              "start_monotonic_ns": started_ns, "end_monotonic_ns": time.monotonic_ns()},
                   "image_protocol": "render_later_from_this_frozen_physical_state"}
        state = self._env._env.get_sim_state()
        poses = ModelPoseBank.capture(domain.sim.model)
        snapshot = SceneSnapshot.capture(episode=episode, tick=tick, task=self.task_key,
            robot_state=observation["robot_state"], language=self.instruction, sim_state=state,
            body_names=poses.names, body_positions=poses.positions[0], body_quaternions=poses.quaternions[0],
            site_rgba=domain.sim.model.site_rgba,
            sampled=sampled, capture_seconds=(time.monotonic_ns() - started_ns) / 1e9)
        self.last_observation["robot_state"] = observation["robot_state"]
        self.last_sample_timing = {"total_s": (time.monotonic_ns() - started_ns) / 1e9,
                                   "camera_render_calls": 0, "scene_sha256": snapshot.scene_sha256}
        return snapshot

    def reference_observation(self, tick):
        self.focus()
        self._defer_cameras = False
        try:
            return super().sample_observation(tick)
        finally:
            self._defer_cameras = True

    def install_images(self, images, sampled):
        self._rendered_images = {key: images[key] for key in ("image", "image2")}
        for key in ("image", "image2"):
            self.last_observation["pixels"][key] = images[key]
            self._sampling[key] = dict(sampled[key])


class SceneRenderer:
    def __init__(self, factory=StateOnlyEnv):
        self.environments, self.factory = {}, factory
        self.instance_id = f"{os.getpid()}-{time.monotonic_ns()}"

    def environment(self, task):
        if task not in self.environments:
            from ..latency_v2.storage import publish
            from .layout import RUN, RUN_ID, budget

            target = RUN / "construction/renderer" / self.instance_id / task
            with budget().track(f"renderer:{task}", "renderer_construction", target,
                                {"task": task, "role": "render_replica"}, run_id=RUN_ID):
                self.environments[task] = self.factory(task_key=task)
                publish(target / "result.json", {"task": task, "constructed": True, "explicit_resets": 0})
        return self.environments[task]

    def render(self, snapshot):
        snapshot.validate()
        env = self.environment(snapshot.task)
        env.focus()
        underlying = env._env._env
        model = underlying.sim.model
        if not np.array_equal(snapshot.body_names, body_names(model)):
            raise ValueError("render replica body layout differs from the frozen scene")
        model.body_pos[:] = snapshot.body_positions
        model.body_quat[:] = snapshot.body_quaternions
        if model.site_rgba.shape != snapshot.site_rgba.shape:
            raise ValueError("render replica site layout differs from the frozen scene")
        model.site_rgba[:] = snapshot.site_rgba
        underlying.set_state(snapshot.sim_state)
        underlying.sim.forward()
        if (env.state_identity() != snapshot.scene_sha256
                or digest_arrays({"site_rgba": model.site_rgba}) != snapshot.visual_sha256):
            raise ValueError("render replica differs from the frozen control scene")
        env._tick = snapshot.tick
        env._sampling = {}
        observation = env.reference_observation(snapshot.tick)
        if (env.state_identity() != snapshot.scene_sha256
                or digest_arrays({"site_rgba": model.site_rgba}) != snapshot.visual_sha256):
            raise ValueError("rendering advanced or changed the frozen physical scene")
        for camera in ("image", "image2"):
            metadata = observation["sampling"][camera]
            if metadata["tick"] != snapshot.tick or metadata["sim_time"] != snapshot.sampled["state"]["sim_time"]:
                raise ValueError("rendered camera belongs to a different physical scene time")
            if observation["pixels"][camera].shape != (256, 256, 3):
                raise ValueError("rendered input resolution changed")
            metadata["physical_state_tick"] = snapshot.tick
            metadata["rendered_in_worker"] = True
        return observation

    def close(self):
        for env in self.environments.values():
            env.focus()
            env.close()
        self.environments.clear()
