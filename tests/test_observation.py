import unittest

import numpy as np

from server.observation import build_observation, get_observation_size


class ObservationTests(unittest.TestCase):
    def test_large_tile_grid_is_cropped_to_configured_size(self):
        config = {
            "normalization": {
                "grid_size": 3,
                "tile_categories_count": 16,
            }
        }
        state = {
            "tile_grid": [[15, 15, 15, 15] for _ in range(4)],
        }

        observation = build_observation(state, config)

        self.assertEqual(observation.shape, (get_observation_size(config),))
        self.assertEqual(observation[13:22].tolist(), [1.0] * 9)

    def test_sprite_hitbox_dimensions_are_exposed_and_inactive_slots_are_zeroed(self):
        config = {
            "normalization": {
                "grid_size": 1,
                "tile_categories_count": 16,
                "max_sprite_hitbox_dimension": 128,
            }
        }
        state = {
            "tile_grid": [[0]],
            "sprites": [
                {
                    "active": 1,
                    "id": 0x9F,
                    "screen_x": 64,
                    "screen_y": 112,
                    "speed_x": -24,
                    "speed_y": 0,
                    "hitbox_width": 52,
                    "hitbox_height": 46,
                },
                {
                    "active": 0,
                    "id": 0x9F,
                    "screen_x": 64,
                    "screen_y": 112,
                    "hitbox_width": 52,
                    "hitbox_height": 46,
                },
            ],
        }

        observation = build_observation(state, config)
        sprites_start = 14

        self.assertEqual(observation.shape, (get_observation_size(config),))
        self.assertAlmostEqual(observation[sprites_start + 6], 52 / 128)
        self.assertAlmostEqual(observation[sprites_start + 7], 46 / 128)
        self.assertEqual(observation[sprites_start + 9:sprites_start + 18].tolist(), [0.0] * 9)

    def test_short_ragged_grid_keeps_legacy_row_order_and_zero_padding(self):
        config = {
            "normalization": {
                "grid_size": 3,
                "tile_categories_count": 16,
            }
        }
        state = {"tile_grid": [[15, 0], [3], [6, 9, 12]]}

        observation = build_observation(state, config)

        np.testing.assert_allclose(
            observation[13:22],
            [1.0, 0.0, 0.2, 0.4, 0.6, 0.8, 0.0, 0.0, 0.0],
        )


if __name__ == "__main__":
    unittest.main()
