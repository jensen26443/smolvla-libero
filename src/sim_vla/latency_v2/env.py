"""The same LIBERO physics and observations, with explicit resets only."""
from __future__ import annotations

import time

import numpy as np

from ..latency_eval.env_adapter import LiberoLatencyEnv
from .model_poses import ModelPoseBank, scene_digest


class ControlledEnv(LiberoLatencyEnv):
    def __init__(self, *, profile_env: bool = False, observation_mode: str = "per_tick", **kwargs) -> None:
        if observation_mode not in {"per_tick", "on_request"}:
            raise ValueError("unsupported observation mode")
        super().__init__(**kwargs)
        self.observation_mode = observation_mode
        self._tick = 0
        self._sampling_tick = 0
        self._sampling = {}
        self._sampling_cameras = True
        self.last_sample_timing = {}
        self.last_observation: dict | None = None
        self.model_pose_bank: ModelPoseBank | None = None
        self.last_step_timing = {}
        self.timing = None
        if profile_env:
            from .env_timing import EnvTiming

            self.timing = EnvTiming()

    def _attach_sampling(self):
        from functools import wraps

        domain = self._env._env.env
        cameras = {"agentview_image": "image", "robot0_eye_in_hand_image": "image2"}
        original = domain._update_observables
        if getattr(original, "_latency_camera_owner", None) is not self:
            @wraps(original)
            def update(force=False):
                if self.observation_mode == "per_tick" or self._sampling_cameras or force:
                    return original(force=force)
                # Same ordered scalar observable updates and model timestep as upstream.
                # Camera observables stay enabled and retain their actual cached values.
                for name, observable in domain._observables.items():
                    if name not in cameras:
                        observable.update(timestep=domain.model_timestep, obs_cache=domain._obs_cache, force=force)

            update._latency_camera_owner = self
            domain._update_observables = update
        for name, key in cameras.items():
            observable = domain._observables[name]
            sensor = observable._sensor
            if getattr(sensor, "_latency_sample_owner", None) is self:
                continue

            def wrap_sensor(original_sensor, camera_key):
                @wraps(original_sensor)
                def sampled(cache):
                    start = time.monotonic_ns()
                    sim_time = float(domain.sim.data.time)
                    image = original_sensor(cache)
                    self._sampling[camera_key] = {"tick": self._sampling_tick,
                        "start_monotonic_ns": start, "end_monotonic_ns": time.monotonic_ns(),
                        "sim_time": sim_time}
                    return image

                sampled._latency_sample_owner = self
                return sampled

            observable._sensor = wrap_sensor(sensor, key)

    def _attach_timing(self):
        domain = self._env._env.env
        self.timing.wrap(domain.sim, "forward", "physics_forward")
        self.timing.wrap(domain.sim, "step", "physics_step")
        self.timing.wrap(domain, "_pre_action", "controller")
        self.timing.wrap(domain, "_update_observables", "other_observables")
        self.timing.wrap(domain, "_get_observations", "observation_collection")
        self.timing.wrap(domain, "_post_action", "post_action_success")
        self.timing.wrap(self._env, "_format_raw_obs", "observation_format")
        for name, observable in domain._observables.items():
            if name in ("agentview_image", "robot0_eye_in_hand_image"):
                self.timing.wrap(observable, "_sensor", "render_" + name)

    def start_episode(self, init_index: int, seed: int, states: np.ndarray | None = None) -> dict:
        self._sampling_cameras = True
        self._tick, self._sampling_tick, self._sampling = 0, 0, {}
        bank = self._env._init_states if states is None else states
        if not 0 <= init_index < len(bank):
            raise IndexError("physical initialization index out of bounds; modulo is forbidden")
        if states is None:
            observation = super().start_episode(init_index, seed)
        else:
            from lerobot.envs.libero import get_libero_dummy_action

            if self.model_pose_bank is None:
                raise ValueError("custom physical state requires its frozen model fixture poses")
            self._env._env.seed(int(seed))
            self._env._env.reset()
            self.model_pose_bank.restore(self._env._env.sim.model, init_index)
            raw = self._env._env.set_init_state(np.asarray(states[init_index], dtype=np.float64))
            for _ in range(self._env.num_steps_wait):
                raw, _, _, _ = self._env._env.step(get_libero_dummy_action())
            for robot in self._env._env.robots:
                robot.controller.use_delta = True
            observation = self._env._format_raw_obs(raw)
            self.n_resets += 1
            self._current_init_index = init_index
        self.last_observation = observation
        self._attach_sampling()
        if self.timing:
            self.timing.take(0.)  # Reset/settle is outside measured control cycles.
            self._attach_timing()  # LIBERO hard resets replace sim and camera observables.
        return observation

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        started = time.monotonic()
        self._sampling_cameras = self.observation_mode == "per_tick"
        self._sampling_tick = self._tick + 1
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (7,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1.0000001):
            raise ValueError("invalid control action")
        raw, reward, done, info = self._env._env.step(action)
        success = bool(self._env._env.check_success())
        info.update({"is_success": success, "done": bool(done), "task_id": self.task_id})
        observation = self._env._format_raw_obs(raw)
        self._tick += 1
        self._sampling["state"] = {"tick": self._tick, "end_monotonic_ns": time.monotonic_ns(),
                                   "sim_time": float(self._env._env.sim.data.time)}
        observation["sampling"] = {key: dict(value) for key, value in self._sampling.items()}
        self.last_observation = observation
        if self.timing:
            self.last_step_timing = self.timing.take(time.monotonic()-started)
        else:
            self.last_step_timing = {"total_s": time.monotonic()-started}
        return observation, float(reward), bool(done or success), False, info

    def sample_observation(self, tick: int) -> dict:
        if tick != self._tick:
            raise ValueError("request tick differs from the current physical boundary")
        started = time.monotonic()
        start_ns = time.monotonic_ns()
        domain = self._env._env.env
        self._sampling_tick, self._sampling_cameras = tick, True
        # Refresh kinematics at the current state without advancing physics.
        domain.sim.forward()
        raw = domain._get_observations(force_update=True)
        observation = self._env._format_raw_obs(raw)
        self._sampling["state"] = {"tick": tick, "start_monotonic_ns": start_ns,
            "end_monotonic_ns": time.monotonic_ns(), "sim_time": float(domain.sim.data.time)}
        if set(self._sampling) != {"image", "image2", "state"}:
            raise RuntimeError("both cameras and state must be sampled for every request")
        if len({value["sim_time"] for value in self._sampling.values()}) != 1:
            raise RuntimeError("request sensors do not correspond to the same physical state")
        observation["sampling"] = {key: dict(value) for key, value in self._sampling.items()}
        self.last_observation = observation
        self.last_sample_timing = self.timing.take(time.monotonic()-started) if self.timing else {
            "total_s": time.monotonic()-started}
        return observation

    def state_identity(self) -> str:
        return scene_digest(np.asarray(self._env._env.get_sim_state()), self._env._env.sim.model)

    def cached_frame(self) -> np.ndarray:
        return self.last_observation["pixels"]["image"].copy()

    def diagnostic(self) -> dict:
        # Evaluator-only: never pass object state/contact telemetry to the scheduler or model.
        domain = self._env._env.env
        from ..tasks import TASK_BY_KEY

        name = TASK_BY_KEY[self.task_key].target_object
        body = domain.sim.model.body_name2id(name + "_main")
        return {"object_position": domain.sim.data.body_xpos[body].tolist(),
                "eef_position": self.last_observation["robot_state"]["eef"]["pos"].tolist()}
