"""Check the sampled physical scene before exposing it to the policy."""
from ..latency_v2.env import ControlledEnv


class VerifiedPhysicalEnv(ControlledEnv):
    def __init__(self, *, expected_scenes, **kwargs):
        super().__init__(**kwargs)
        self.expected_scenes = tuple(expected_scenes)

    def start_episode(self, init_index, seed, states=None):
        observation = super().start_episode(init_index, seed, states=states)
        if states is None or self.state_identity() != self.expected_scenes[init_index]:
            raise ValueError("restored physical scene differs from its policy-blind generation evidence")
        return observation
