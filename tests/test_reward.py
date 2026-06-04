import unittest

from server.reward import EpisodeTracker, compute_reward


def state(**overrides) -> dict:
    result = {
        "mario_x_level": 0,
        "mario_y_level": 0,
        "coins": 0,
        "powerup": 0,
        "lives": 0,
        "player_anim": 0,
        "sublevel": 0,
        "is_vertical": False,
        "sprites": [],
        "step": 1,
    }
    result.update(overrides)
    return result


def config() -> dict:
    return {
        "rewards": {
            "enemy_killed": 2.0,
            "time_penalty_per_step": 0.0,
        },
        "ppo": {
            "stagnation_timeout_steps": 999,
            "max_episode_steps": 999,
        },
    }


class RewardTests(unittest.TestCase):
    def test_sprite_death_state_produces_kill_reward(self):
        tracker = EpisodeTracker(is_first_step=False)
        tracker.last_sprite_status[0] = 0x08

        reward, event, done = compute_reward(
            state(sprites=[{"status": 0x02}]), tracker, config()
        )

        self.assertEqual(reward, 2.0)
        self.assertIn("KILL x1", event)
        self.assertFalse(done)

    def test_goal_sets_success_flag(self):
        tracker = EpisodeTracker(is_first_step=False)

        reward, event, done = compute_reward(
            state(player_anim=0x0A), tracker, config()
        )

        self.assertEqual(reward, 1000.0)
        self.assertIn("GOAL", event)
        self.assertTrue(done)
        self.assertTrue(tracker.goal_reached)

    def test_horizontal_progress_cannot_be_farmed_by_backtracking(self):
        tracker = EpisodeTracker(is_first_step=False, max_x=100, last_x=100)
        reward_config = config()
        reward_config["rewards"]["x_progress_per_pixel"] = 0.3

        reward, _, _ = compute_reward(
            state(mario_x_level=110), tracker, reward_config
        )
        self.assertAlmostEqual(reward, 3.0)

        reward, _, _ = compute_reward(
            state(mario_x_level=90), tracker, reward_config
        )
        self.assertEqual(reward, 0.0)

        reward, _, _ = compute_reward(
            state(mario_x_level=110), tracker, reward_config
        )
        self.assertEqual(reward, 0.0)

    def test_exploration_reward_is_granted_once_per_cell(self):
        tracker = EpisodeTracker(
            is_first_step=False,
            max_x=100,
            last_x=100,
            visited_cells={(0, 6, 0)},
        )
        reward_config = config()
        reward_config["rewards"]["exploration_new_cell"] = 0.5

        reward, event, _ = compute_reward(
            state(mario_x_level=90), tracker, reward_config
        )
        self.assertEqual(reward, 0.5)
        self.assertIn("EXPLORE", event)

        reward, event, _ = compute_reward(
            state(mario_x_level=90), tracker, reward_config
        )
        self.assertEqual(reward, 0.0)
        self.assertNotIn("EXPLORE", event)

    def test_exploration_in_horizontal_level_resets_stagnation(self):
        tracker = EpisodeTracker(
            is_first_step=False,
            steps_without_progress=2,
            visited_cells={(0, 0, 0)},
        )
        reward_config = config()
        reward_config["rewards"]["exploration_new_cell"] = 0.5
        reward_config["ppo"]["stagnation_timeout_steps"] = 3

        _, event, done = compute_reward(
            state(mario_y_level=16), tracker, reward_config
        )

        self.assertNotIn("TIMEOUT", event)
        self.assertFalse(done)
        self.assertEqual(tracker.steps_without_progress, 0)

    def test_stagnation_timeout_is_marked_as_truncation(self):
        tracker = EpisodeTracker(is_first_step=False, steps_without_progress=2)
        reward_config = config()
        reward_config["ppo"]["stagnation_timeout_steps"] = 3

        _, event, done = compute_reward(state(), tracker, reward_config)

        self.assertIn("TIMEOUT", event)
        self.assertTrue(done)
        self.assertTrue(tracker.truncated)


if __name__ == "__main__":
    unittest.main()
