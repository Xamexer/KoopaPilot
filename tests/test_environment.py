import unittest

from server.environment import DISCRETE_ACTIONS


class EnvironmentTests(unittest.TestCase):
    def test_jump_actions_do_not_press_up(self):
        for action_index in [2, 3, 5, 6, 7, 8]:
            with self.subTest(action_index=action_index):
                self.assertEqual(DISCRETE_ACTIONS[action_index][2], 0)

    def test_door_and_climb_action_still_presses_up(self):
        self.assertEqual(DISCRETE_ACTIONS[10][2], 1)


if __name__ == "__main__":
    unittest.main()
