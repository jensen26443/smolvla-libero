"""A copied physical scene; cameras are resolved only by the render owner."""
from dataclasses import dataclass

import numpy as np

from ..latency_v2.schema import digest_arrays, frozen_copy


@dataclass(frozen=True)
class SceneSnapshot:
    episode: str
    tick: int
    task: str
    robot_state: dict
    language: str
    sim_state: np.ndarray
    body_names: np.ndarray
    body_positions: np.ndarray
    body_quaternions: np.ndarray
    site_rgba: np.ndarray
    sampled: dict
    capture_seconds: float
    digest: str
    scene_sha256: str
    visual_sha256: str
    return_images: bool = True

    @property
    def images(self):
        return {}

    def packet(self):
        raise RuntimeError("scene snapshot must be restored and rendered before model preprocessing")

    def validate(self):
        if self.tick < 0 or self.sim_state.ndim != 1 or not np.isfinite(self.sim_state).all():
            raise ValueError("invalid frozen simulator state")
        n = len(self.body_names)
        if (self.body_names.ndim != 1 or self.body_positions.shape != (n, 3)
                or self.body_quaternions.shape != (n, 4) or not np.isfinite(self.body_positions).all()
                or not np.isfinite(self.body_quaternions).all()
                or not np.allclose(np.linalg.norm(self.body_quaternions, axis=-1), 1., atol=1e-6, rtol=0)):
            raise ValueError("invalid frozen model body layout or transforms")
        if (self.site_rgba.ndim != 2 or self.site_rgba.shape[1] != 4
                or not np.isfinite(self.site_rgba).all()):
            raise ValueError("invalid frozen model site visualization")
        scene = digest_arrays({"state": self.sim_state, "body_pos": self.body_positions,
                               "body_quat": self.body_quaternions})
        visual = digest_arrays({"site_rgba": self.site_rgba})
        digest = digest_arrays({"scene": scene, "visual": visual, "body_names": self.body_names, "robot_state": self.robot_state,
                                "task": self.task, "tick": self.tick, "language": self.language})
        if scene != self.scene_sha256 or visual != self.visual_sha256 or digest != self.digest:
            raise ValueError("frozen scene or model input changed")
        if self.sampled.get("state", {}).get("tick") != self.tick:
            raise ValueError("frozen state does not belong to the request physical tick")

    @classmethod
    def capture(cls, *, episode, tick, task, robot_state, language, sim_state, body_names,
                body_positions, body_quaternions, site_rgba, sampled, capture_seconds=0.):
        state = frozen_copy(np.asarray(sim_state, dtype="<f8"))
        positions = frozen_copy(np.asarray(body_positions, dtype="<f8"))
        quaternions = frozen_copy(np.asarray(body_quaternions, dtype="<f8"))
        robot = frozen_copy(robot_state)
        sites = frozen_copy(np.asarray(site_rgba, dtype="<f4"))
        scene = digest_arrays({"state": state, "body_pos": positions, "body_quat": quaternions})
        visual = digest_arrays({"site_rgba": sites})
        digest = digest_arrays({"scene": scene, "visual": visual, "body_names": np.asarray(body_names),
                                "robot_state": robot, "task": task, "tick": tick,
                                "language": language})
        result = cls(episode, tick, task, robot, language, state, frozen_copy(np.asarray(body_names)),
                     positions, quaternions, sites, frozen_copy(sampled), capture_seconds, digest, scene, visual)
        result.validate()
        return result
