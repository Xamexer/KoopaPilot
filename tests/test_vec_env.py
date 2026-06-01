import unittest

import numpy as np
from gymnasium import spaces

from server.vec_env import SMWVecEnv


class TruncatingEnvironment:
    def __init__(self):
        self.config = {"normalization": {"grid_size": 1}}
        self.render_mode = None
        self.observation_space = spaces.Box(
            low=-0.5, high=1.5, shape=(122,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(1)
        self.reset_count = 0

    def step(self, action):
        observation = np.ones(122, dtype=np.float32)
        return observation, 1.0, False, True, {}

    def reset(self):
        self.reset_count += 1
        return np.zeros(122, dtype=np.float32), {}


class VectorEnvironmentTests(unittest.TestCase):
    def test_truncation_is_forwarded_for_sb3_bootstrapping(self):
        environment = TruncatingEnvironment()
        vec_env = SMWVecEnv([environment])

        vec_env.step_async(np.array([0]))
        observation, reward, done, info = vec_env.step_wait()

        self.assertTrue(done[0])
        self.assertTrue(info[0]["TimeLimit.truncated"])
        self.assertEqual(info[0]["terminal_observation"].tolist(), [1.0] * 122)
        self.assertEqual(observation[0].tolist(), [0.0] * 122)
        self.assertEqual(reward[0], 1.0)
        self.assertEqual(environment.reset_count, 1)


if __name__ == "__main__":
    unittest.main()
