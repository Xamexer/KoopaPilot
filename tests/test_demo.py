import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from server.demo import DemoManager, _demo_config, _find_latest_model_path
from server.agent import _save_model_atomic
from server.main import _backend_override_for_mode, _live_model_path


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

    def test_live_model_mirror_never_aliases_resume_source(self):
        model_dir = os.path.abspath("models")
        resume_path = os.path.join(model_dir, "model_live.zip")

        self.assertEqual(
            _live_model_path(model_dir, resume_path),
            os.path.join(model_dir, "model_live_viewer.zip"),
        )

        self.assertEqual(
            _live_model_path(model_dir, os.path.join(model_dir, "MODEL_LIVE")),
            os.path.join(model_dir, "model_live_viewer.zip"),
        )

    def test_modes_choose_the_backend_they_actually_use(self):
        self.assertEqual(_backend_override_for_mode("live-demo", "bizhawk"), "retrojet")
        self.assertEqual(_backend_override_for_mode("demo", "retrojet"), "bizhawk")
        self.assertEqual(_backend_override_for_mode("training", "retrojet"), "retrojet")
        self.assertIsNone(_backend_override_for_mode("training", None))

    def test_live_model_does_not_reload_mid_episode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model_live.zip"
            model_path.write_bytes(b"new model")
            old_model = object()
            manager = DemoManager(
                vec_env=SimpleNamespace(),
                config={"paths": {}},
                model_path=str(model_path),
                reload_on_change=True,
            )
            manager.model = old_model
            manager.model_mtime_ns = 0

            with patch("stable_baselines3.PPO.load") as load:
                self.assertTrue(manager._ensure_model(allow_reload=False))

            self.assertIs(manager.model, old_model)
            load.assert_not_called()

    def test_viewer_load_and_mirror_replace_are_serialized(self):
        class FakeWriterModel:
            @staticmethod
            def save(path):
                Path(path).write_bytes(b"replacement")

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model_live.zip"
            model_path.write_bytes(b"initial")
            manager = DemoManager(
                vec_env=SimpleNamespace(),
                config={"paths": {}},
                model_path=str(model_path),
                reload_on_change=True,
            )
            load_started = threading.Event()
            release_load = threading.Event()
            writer_finished = threading.Event()

            def fake_load(*args, **kwargs):
                load_started.set()
                self.assertTrue(release_load.wait(2))
                return SimpleNamespace(num_timesteps=0)

            reader = threading.Thread(target=manager._ensure_model)

            def write_mirror():
                _save_model_atomic(FakeWriterModel(), str(model_path), 1)
                writer_finished.set()

            with patch("stable_baselines3.PPO.load", side_effect=fake_load):
                reader.start()
                self.assertTrue(load_started.wait(2))
                writer = threading.Thread(target=write_mirror)
                writer.start()
                self.assertFalse(writer_finished.wait(0.05))
                release_load.set()
                reader.join(2)
                writer.join(2)

            self.assertFalse(reader.is_alive())
            self.assertFalse(writer.is_alive())
            self.assertTrue(writer_finished.is_set())

    def test_demo_reports_deterministic_episode_to_callback(self):
        vec_env = SimpleNamespace(
            num_envs=1,
            reset=Mock(return_value=np.zeros((1, 2), dtype=np.float32)),
            step=Mock(return_value=(
                np.zeros((1, 2), dtype=np.float32),
                np.array([12.0]),
                np.array([True]),
                [{"episode": {
                    "r": 12.0,
                    "l": 3,
                    "max_x": 456,
                    "goal_reached": True,
                }}],
            )),
        )
        callback = Mock()
        manager = DemoManager(
            vec_env=vec_env,
            config={"paths": {}},
            model_path="model.zip",
            episode_callback=callback,
        )
        manager.model = SimpleNamespace(
            predict=Mock(return_value=(np.array([1]), None))
        )
        manager.model_timestep = 4096

        with patch.object(manager, "_ensure_model", return_value=True):
            manager.run(episodes=1)

        callback.assert_called_once_with({
            "reward": 12.0,
            "length": 3,
            "max_x": 456.0,
            "goal_reached": True,
            "checkpoint_timestep": 4096,
        })


if __name__ == "__main__":
    unittest.main()
