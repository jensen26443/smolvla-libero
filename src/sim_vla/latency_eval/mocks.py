"""Deterministic mock environment and policy for timing-semantic tests.

The mocks exist only to verify code correctness (PLAN-03 §8); their results are
never substitutes for real VLA experiments. Both are fully deterministic:
identical inputs and seeds produce identical traces, and every hidden state
the evaluator sees is kept out of the policy-visible channels.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .contracts import ActionSpec, ObservationPacket

MOCK_IMAGE_SIZE = 32
# Distinct gray levels let tests script frame differences deterministically.
BLACK_FRAME = np.zeros((MOCK_IMAGE_SIZE, MOCK_IMAGE_SIZE, 3), dtype=np.uint8)
GRAY_FRAME = np.full((MOCK_IMAGE_SIZE, MOCK_IMAGE_SIZE, 3), 128, dtype=np.uint8)
WHITE_FRAME = np.full((MOCK_IMAGE_SIZE, MOCK_IMAGE_SIZE, 3), 255, dtype=np.uint8)


class MockEnv:
    """Toy environment: step counter, scripted frames, hidden truth.

    Policy-visible channels: ``pixels.image`` encodes the step parity/level and
    a state vector. Evaluator-only channels (``secret_object_pos``) live in
    ``info`` and never appear in the observation dict. Success fires when the
    executed action's first component exceeds ``success_threshold`` for
    ``success_burst`` consecutive ticks (a controllable, non-trivial predicate).
    """

    def __init__(
        self,
        *,
        instruction: str = "mock task",
        frame_schedule: list[np.ndarray] | None = None,
        success_threshold: float = 0.9,
        success_burst: int = 3,
        max_steps: int = 300,
    ) -> None:
        self.instruction = instruction
        self.frame_schedule = frame_schedule or [BLACK_FRAME, GRAY_FRAME, WHITE_FRAME]
        self.success_threshold = success_threshold
        self.success_burst = success_burst
        self.max_steps = max_steps
        self.n_steps = 0
        self.n_resets = 0
        self.step_actions: list[np.ndarray] = []
        self.hidden_truth: float = 0.0  # evaluator-only; mutates without obs change
        self._burst_count = 0
        self._current_obs: dict[str, Any] | None = None

    def _obs(self) -> dict[str, Any]:
        frame = self.frame_schedule[min(self.n_steps, len(self.frame_schedule) - 1)]
        return {
            "pixels": {"image": frame.copy(), "image2": frame.copy()},
            "robot_state": {
                "eef": {
                    "pos": np.array([float(self.n_steps), 0.0, 0.0]),
                    "quat": np.array([1.0, 0.0, 0.0, 0.0]),
                },
                "gripper": {"qpos": np.array([0.05, 0.05])},
            },
        }

    def start_episode(self, init_index: int, seed: int) -> dict[str, Any]:
        self.n_steps = 0
        self.n_resets += 1
        self.step_actions = []
        self._burst_count = 0
        self._current_obs = self._obs()
        return self._current_obs

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.n_steps += 1
        self.step_actions.append(np.array(action, copy=True))
        if float(action[0]) > self.success_threshold:
            self._burst_count += 1
        else:
            self._burst_count = 0
        is_success = self._burst_count >= self.success_burst
        terminated = is_success or self.n_steps >= self.max_steps
        info: dict[str, Any] = {
            "is_success": is_success,
            "step": self.n_steps,
            # Evaluator-only channel: changes never propagate into _obs().
            "secret_object_pos": np.array([self.hidden_truth, 0.0, 0.0]),
        }
        self._current_obs = self._obs()
        return self._current_obs, float(is_success), terminated, False, info


class MockPolicy:
    """Scripted chunk policy derived only from its (frozen) input packet.

    ``mode="capture_echo"``: chunk rows equal ``[capture_tick, i, packet_id]``
    padded to ``dim`` — so any leakage of a later-mutated cache into the input
    changes the output and the snapshot test catches it.
    ``mode="push_right"``: constant strong actions that drive MockEnv success.
    """

    def __init__(
        self,
        *,
        horizon: int = 50,
        dim: int = 7,
        mode: str = "capture_echo",
        checkpoint_label: str = "mock-policy",
    ) -> None:
        self.horizon = horizon
        self.dim = dim
        self.mode = mode
        self.checkpoint_label = checkpoint_label
        self.chunk_size = horizon
        self.calls: list[int] = []  # packet ids actually consumed
        self.n_resets = 0

    def reset(self) -> None:
        self.n_resets += 1
        self.calls = self.calls  # deliberate no-op beyond counting; no hidden history

    def set_episode_seed(self, seed: int) -> None:
        return

    def infer(self, packet: ObservationPacket, request_id: int, request_tick: int) -> np.ndarray:
        self.calls.append(packet.packet_id)
        chunk = np.zeros((self.horizon, self.dim), dtype=np.float64)
        if self.mode == "capture_echo":
            for i in range(self.horizon):
                chunk[i, 0] = float(packet.capture_tick % 1000) / 1000.0
                chunk[i, 1] = float(i) / 100.0
                chunk[i, 2] = float(packet.packet_id % 100) / 100.0
        elif self.mode == "push_right":
            chunk[:, 0] = 1.0
        else:
            raise ValueError(f"unknown mock mode {self.mode}")
        return chunk


MOCK_ACTION_SPEC = ActionSpec(horizon=50)
