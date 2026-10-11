"""Real LIBERO environment adapter for the latency runner.

Wraps a single (non-vectorized) lerobot ``LiberoEnv`` with explicit
``init_state_id`` control per episode. ``AutoresetMode`` footguns do not apply
because we call ``step`` only while the episode is live; the env-internal
auto-reset after termination is harmless because every episode sets
``init_state_id`` explicitly before reset. Heavy imports are deferred so the
lightweight default install can still import this package for tests.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ..assets import libero_assets_dir, project_root
from ..runtime import import_libero
from ..tasks import TASK_BY_KEY, TaskSpec


class LiberoLatencyEnv:
    """One LIBERO Goal task behind the runner's env protocol."""

    def __init__(
        self,
        *,
        task_key: str,
        episode_length: int = 300,
        observation_width: int = 256,
        observation_height: int = 256,
        device_id: int = 0,
    ) -> None:
        task_spec: TaskSpec = TASK_BY_KEY[task_key]
        self.task_key = task_key
        self.task_id = task_spec.suite_task_id
        import_libero(root=project_root(), assets=libero_assets_dir())
        from lerobot.envs.libero import LiberoEnv
        from libero.libero import benchmark

        suite = benchmark.get_benchmark_dict()[task_spec.suite]()
        suite_task = suite.get_task(task_spec.suite_task_id)
        self.instruction = str(suite_task.language).lower()
        self._env = LiberoEnv(
            task_suite=suite,
            task_id=task_spec.suite_task_id,
            task_suite_name=task_spec.suite,
            episode_length=episode_length,
            camera_name="agentview_image,robot0_eye_in_hand_image",
            obs_type="pixels_agent_pos",
            observation_width=observation_width,
            observation_height=observation_height,
            init_states=True,
            episode_index=0,
            n_envs=1,
            control_mode="relative",
        )
        self.episode_length = episode_length
        self.n_resets = 0
        self._current_init_index: int | None = None

    def start_episode(self, init_index: int, seed: int) -> dict[str, Any]:
        """Reset to an explicitly chosen initialization under an explicit seed.

        ``LiberoEnv.reset`` consumes ``init_state_id`` then advances it by the
        reset stride; setting it here right before reset pins the exact init
        row regardless of any internal auto-reset in between episodes.
        """

        self._env.init_state_id = int(init_index)
        observation, _info = self._env.reset(seed=int(seed))
        self.n_resets += 1
        self._current_init_index = int(init_index)
        return observation

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        return self._env.step(np.asarray(action, dtype=np.float64))

    @property
    def current_init_index(self) -> int | None:
        return self._current_init_index

    def get_sim_state_digest(self) -> str:
        import hashlib

        state = np.asarray(self._env._env.get_sim_state(), dtype=np.float64)
        return hashlib.sha256(state.tobytes()).hexdigest()

    def render_frame(self) -> np.ndarray | None:
        frame = self._env.render()
        return None if frame is None else np.asarray(frame)

    def close(self) -> None:
        self._env.close()


RenderCallback = Callable[[int], None]


class VideoRecorder:
    """Collects rendered frames for sampled episodes and writes an MP4.

    Only trials selected for video sampling instantiate this; the main loop
    overhead for everyone else is a None check.
    """

    def __init__(self, env: LiberoLatencyEnv, *, fps: int = 20) -> None:
        self.env = env
        self.fps = fps
        self.frames: list[np.ndarray] = []

    def reset(self) -> None:
        self.frames = []

    def capture(self, tick: int) -> None:
        frame = self.env.render_frame()
        if frame is not None:
            self.frames.append(frame)

    def save(self, path: str | Path) -> Path:
        import imageio.v2 as imageio

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(target, self.frames, fps=self.fps, quality=8, macro_block_size=1)
        return target
