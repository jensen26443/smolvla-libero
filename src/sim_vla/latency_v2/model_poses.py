"""LIBERO fixture placements live in the MuJoCo model, outside flattened state."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .schema import digest_arrays


def body_names(model) -> np.ndarray:
    return np.asarray([model.body_id2name(index) or "" for index in range(model.nbody)], dtype=str)


def scene_digest(state: np.ndarray, model) -> str:
    # Fixed fixture transforms affect observations, contacts, and success predicates.
    return digest_arrays({"state": np.asarray(state, dtype="<f8"),
                          "body_pos": np.asarray(model.body_pos, dtype="<f8"),
                          "body_quat": np.asarray(model.body_quat, dtype="<f8")})


@dataclass(frozen=True)
class ModelPoseBank:
    names: np.ndarray
    positions: np.ndarray
    quaternions: np.ndarray

    @classmethod
    def capture(cls, model) -> ModelPoseBank:
        return cls(body_names(model), np.asarray(model.body_pos, dtype=np.float64)[None].copy(),
                   np.asarray(model.body_quat, dtype=np.float64)[None].copy())

    @classmethod
    def load(cls, path: Path, *, expected_states: int) -> ModelPoseBank:
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != {"body_names", "body_pos", "body_quat"}:
                raise ValueError("model pose asset requires body names, positions, and quaternions")
            bank = cls(arrays["body_names"].copy(), arrays["body_pos"].copy(), arrays["body_quat"].copy())
        bank.validate(expected_states)
        return bank

    def validate(self, expected_states: int) -> None:
        if self.names.ndim != 1 or self.names.dtype.kind != "U" or len(self.names) == 0:
            raise ValueError("invalid model body name array")
        if self.positions.shape != (expected_states, len(self.names), 3):
            raise ValueError("model body positions do not match state bank")
        if self.quaternions.shape != (expected_states, len(self.names), 4):
            raise ValueError("model body quaternions do not match state bank")
        if not np.isfinite(self.positions).all() or not np.isfinite(self.quaternions).all():
            raise ValueError("non-finite model body transforms")
        if not np.allclose(np.linalg.norm(self.quaternions, axis=-1), 1., atol=1e-6, rtol=0.):
            raise ValueError("model body quaternions must be normalized")

    def restore(self, model, index: int) -> None:
        if not 0 <= index < len(self.positions):
            raise IndexError("model pose index out of bounds; modulo is forbidden")
        if not np.array_equal(self.names, body_names(model)):
            raise ValueError("model body layout differs from physical initialization asset")
        model.body_pos[:] = self.positions[index]
        model.body_quat[:] = self.quaternions[index]

    def save(self, path: Path) -> None:
        self.validate(len(self.positions))
        np.savez(path, body_names=self.names, body_pos=self.positions, body_quat=self.quaternions)
