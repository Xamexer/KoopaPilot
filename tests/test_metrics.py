import json
import tempfile
import unittest
from pathlib import Path

from server.metrics import MetricsLogger


class MetricsTests(unittest.TestCase):
    def test_retrojet_summary_and_deterministic_evaluation_are_recorded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = {
                "_config_dir": str(root),
                "_backend": "retrojet",
                "paths": {
                    "rom": str(root / "roms" / "game.sfc"),
                    "log_dir": str(root / "logs"),
                },
                "emulator": {"num_instances": 8, "frame_skip": 4},
                "backend": {
                    "type": "retrojet",
                    "retrojet": {
                        "core_path": str(root.parent / "RetroJet" / "core.dll"),
                        "num_envs": 32,
                    },
                },
                "level_loading": {
                    "mode": "level_loading",
                    "levels": ["0x105"],
                },
                "ppo": {"n_steps": 128, "batch_size": 256},
                "performance": {
                    "torch_threads": 4,
                    "retrojet_threads": 4,
                },
            }
            metrics = MetricsLogger(str(root / "logs"), config)
            metrics.log_iteration({"timestep": 2048, "mean_reward": 5.0})
            metrics.log_evaluation({
                "reward": 1000.0,
                "length": 200,
                "goal_reached": True,
                "max_x": 4800,
                "checkpoint_timestep": 1024,
            })

            data = json.loads(Path(metrics.metrics_file).read_text())
            snapshot = json.loads(Path(metrics.config_file).read_text())

            self.assertEqual(data["config_summary"]["num_envs"], 32)
            self.assertEqual(data["config_summary"]["frame_skip"], 4)
            self.assertEqual(data["config_summary"]["torch_threads"], 4)
            self.assertEqual(data["config_summary"]["retrojet_threads"], 4)
            self.assertEqual(data["evaluations"][0]["timestep"], 1024)
            self.assertEqual(
                data["evaluations"][0]["action_selection"], "deterministic"
            )
            self.assertEqual(snapshot["paths"]["rom"], "./roms/game.sfc")
            self.assertNotIn("_config_dir", snapshot)

    def test_rapid_runs_get_distinct_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"paths": {}, "emulator": {}, "ppo": {}}
            first = MetricsLogger(temp_dir, config)
            second = MetricsLogger(temp_dir, config)

            self.assertNotEqual(first.run_id, second.run_id)
            self.assertNotEqual(first.run_dir, second.run_dir)


if __name__ == "__main__":
    unittest.main()
