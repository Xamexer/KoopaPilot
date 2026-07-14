import os
import tempfile
import unittest
from pathlib import Path

from server.demo import _demo_config, _find_latest_model_path


class DemoTests(unittest.TestCase):
    def test_demo_config_uses_visible_normal_speed_copy(self):
        config = {
            "paths": {"model_dir": "./models"},
            "emulator": {
                "num_instances": 8,
                "base_port": 9000,
                "speed_percent": 6400,
            },
            "flags": {
                "visibility": False,
                "reward_display": False,
                "button_input_display": False,
            },
        }

        demo = _demo_config(config, num_emulators=2, base_port=10000)

        self.assertEqual(demo["_mode"], "demo")
        self.assertEqual(demo["emulator"]["num_instances"], 2)
        self.assertEqual(demo["emulator"]["base_port"], 10000)
        self.assertEqual(demo["emulator"]["speed_percent"], 100)
        self.assertTrue(demo["flags"]["visibility"])
        self.assertTrue(demo["flags"]["reward_display"])
        self.assertTrue(demo["flags"]["button_input_display"])

        self.assertEqual(config["emulator"]["num_instances"], 8)
        self.assertEqual(config["emulator"]["speed_percent"], 6400)
        self.assertFalse(config["flags"]["visibility"])

    def test_demo_config_clamps_emulator_count(self):
        demo = _demo_config({"emulator": {}, "flags": {}}, 0, None)

        self.assertEqual(demo["emulator"]["num_instances"], 1)

    def test_find_latest_model_path_prefers_newest_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old_checkpoint = root / "model_step_100.zip"
            new_checkpoint = root / "model_step_200.zip"
            best = root / "model_best.zip"
            for path in [old_checkpoint, new_checkpoint, best]:
                path.write_bytes(b"model")

            os.utime(old_checkpoint, (1000, 1000))
            os.utime(best, (1100, 1100))
            os.utime(new_checkpoint, (1200, 1200))

            self.assertEqual(_find_latest_model_path(temp_dir), str(new_checkpoint))

    def test_find_latest_model_path_returns_none_without_models(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertIsNone(_find_latest_model_path(temp_dir))


if __name__ == "__main__":
    unittest.main()
