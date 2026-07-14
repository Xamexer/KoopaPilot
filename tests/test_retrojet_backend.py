import unittest

import numpy as np

from server.retrojet_backend import RetroJetVecEnv


def config() -> dict:
    return {
        "normalization": {
            "grid_size": 1,
            "tile_categories_count": 16,
            "max_sprite_hitbox_dimension": 128,
        },
        "rewards": {
            "x_progress_per_pixel": 1.0,
            "time_penalty_per_step": 0.0,
        },
        "ppo": {
            "stagnation_timeout_steps": 999,
            "max_episode_steps": 999,
        },
    }


def state(x: int) -> dict:
    return {
        "mario_x_screen": x,
        "mario_y_screen": 112,
        "mario_x_level": x,
        "mario_y_level": 112,
        "tile_grid": [[0]],
        "sprites": [],
        "coins": 0,
        "powerup": 0,
        "lives": 0,
        "player_anim": 0,
        "sublevel": 0,
        "is_vertical": False,
    }


class FakeRetroJetRunner:
    num_envs = 2

    def __init__(self):
        self.positions = [0, 10]
        self.reset_calls = []

    def reset_all(self):
        self.positions = [0, 10]
        return [state(x) for x in self.positions]

    def reset(self, indices):
        self.reset_calls.append(list(indices))
        for idx in indices:
            self.positions[idx] = 0
        return [state(self.positions[idx]) for idx in indices]

    def step(self, actions):
        self.positions = [x + 16 for x in self.positions]
        return [state(x) for x in self.positions]


class RetroJetBackendTests(unittest.TestCase):
    def test_reset_builds_observations_for_all_envs(self):
        env = RetroJetVecEnv(config(), runner=FakeRetroJetRunner())

        obs = env.reset()

        self.assertEqual(obs.shape, (2, 122))
        self.assertAlmostEqual(obs[0][0], 0 / 256)
        self.assertAlmostEqual(obs[1][0], 10 / 256)

    def test_step_uses_koopapilot_reward_logic(self):
        runner = FakeRetroJetRunner()
        env = RetroJetVecEnv(config(), runner=runner)
        env.reset()

        env.step_async(np.array([1, 1]))
        obs, rewards, dones, infos = env.step_wait()

        self.assertEqual(obs.shape, (2, 122))
        self.assertEqual(rewards.tolist(), [16.0, 16.0])
        self.assertEqual(dones.tolist(), [False, False])
        self.assertEqual(infos[0]["mario_x"], 16)


if __name__ == "__main__":
    unittest.main()
