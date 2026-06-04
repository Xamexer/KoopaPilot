import threading
import time
import unittest

from server.environment import (
    DISCRETE_ACTIONS,
    NO_OP_RESPONSE,
    SMWEnvironment,
)


def config() -> dict:
    return {
        "normalization": {
            "grid_size": 1,
            "tile_categories_count": 16,
        },
        "rewards": {},
        "ppo": {},
    }


def state(mario_x_screen: int) -> dict:
    return {
        "mario_x_screen": mario_x_screen,
        "mario_y_screen": 112,
        "mario_x_level": mario_x_screen,
        "mario_y_level": 112,
        "tile_grid": [[0]],
        "sprites": [],
    }


def wait_until(predicate, timeout: float = 1.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for condition")
        time.sleep(0.001)


class EnvironmentTests(unittest.TestCase):
    def test_jump_actions_do_not_press_up(self):
        for action_index in [2, 3, 5, 6, 7, 8]:
            with self.subTest(action_index=action_index):
                self.assertEqual(DISCRETE_ACTIONS[action_index][2], 0)

    def test_door_and_climb_action_still_presses_up(self):
        self.assertEqual(DISCRETE_ACTIONS[10][2], 1)

    def test_bootstrap_noop_releases_run_button(self):
        self.assertEqual(NO_OP_RESPONSE["action"], [0, 0, 0, 0, 0, 0, 1])

    def test_first_reset_is_sent_and_waits_for_fresh_state(self):
        environment = SMWEnvironment(emulator_id=0, config=config())
        reset_result = {}

        reset_thread = threading.Thread(
            target=lambda: reset_result.setdefault("value", environment.reset()),
            daemon=True,
        )
        reset_thread.start()
        wait_until(lambda: environment._initialized)

        before_response = {}
        before_thread = threading.Thread(
            target=lambda: before_response.setdefault(
                "value", environment.on_state_received(state(32))
            ),
            daemon=True,
        )
        before_thread.start()
        before_thread.join(timeout=1.0)

        self.assertEqual(before_response["value"], {"type": "reset"})
        self.assertTrue(reset_thread.is_alive())

        after_response = {}
        after_thread = threading.Thread(
            target=lambda: after_response.setdefault(
                "value", environment.on_state_received(state(96))
            ),
            daemon=True,
        )
        after_thread.start()
        reset_thread.join(timeout=1.0)
        after_thread.join(timeout=1.0)

        self.assertFalse(reset_thread.is_alive())
        self.assertEqual(after_response["value"], NO_OP_RESPONSE)
        observation, _ = reset_result["value"]
        self.assertAlmostEqual(observation[0], 96 / 256)


if __name__ == "__main__":
    unittest.main()
