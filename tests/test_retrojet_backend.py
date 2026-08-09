import unittest
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from server.retrojet_backend import RetroJetVecEnv, _create_runner


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
        self.closed = False

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

    def close(self):
        self.closed = True


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

    def test_close_releases_native_runner_immediately_and_is_idempotent(self):
        runner = FakeRetroJetRunner()
        env = RetroJetVecEnv(config(), runner=runner)

        env.close()
        env.close()

        self.assertTrue(runner.closed)
        self.assertIsNone(env.runner)

    def test_runner_uses_shared_rom_level_and_frame_skip_settings(self):
        runner = Mock()
        retrojet_module = SimpleNamespace(Runner=Mock(return_value=runner))
        cfg = config()
        cfg.update({
            "paths": {"rom": "./roms/shared.sfc"},
            "emulator": {"num_instances": 1, "frame_skip": 5},
            "backend": {
                "type": "retrojet",
                "retrojet": {
                    "core_path": "../RetroJet/cores/core.dll",
                    "num_envs": 3,
                    "boot_frames": 120,
                    "savestate_paths": [],
                },
            },
            "level_loading": {
                "mode": "level_loading",
                "levels": ["0x106"],
            },
            "performance": {"retrojet_threads": 4},
        })

        with patch.dict(sys.modules, {"retrojet": retrojet_module}):
            self.assertIs(_create_runner(cfg), runner)

        kwargs = retrojet_module.Runner.call_args.kwargs
        self.assertEqual(kwargs["rom_path"], "roms\\shared.sfc")
        self.assertEqual(kwargs["frame_skip"], 5)
        self.assertEqual(kwargs["level_ids"], [0x106])
        self.assertEqual(kwargs["num_envs"], 3)
        self.assertEqual(kwargs["num_threads"], 4)


if __name__ == "__main__":
    unittest.main()
