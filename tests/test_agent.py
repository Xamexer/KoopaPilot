import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from server.agent import (
    CheckpointCallback,
    _model_metadata_path,
    _replace_with_retry,
    _save_model_atomic,
    create_agent,
)


class AgentTests(unittest.TestCase):
    @patch("server.agent.PPO.load")
    def test_resume_rejects_mismatched_rollout_size(self, load_model):
        load_model.return_value = SimpleNamespace(n_steps=2048)
        config = {
            "paths": {},
            "ppo": {
                "n_steps": 128,
            },
        }

        with tempfile.NamedTemporaryFile() as checkpoint:
            with self.assertRaisesRegex(ValueError, "n_steps=2048"):
                create_agent(None, config, checkpoint.name)
        self.assertEqual(load_model.return_value.tensorboard_log, "./logs")

    def test_checkpoint_metrics_include_recent_goal_rate(self):
        metrics_logger = SimpleNamespace(log_iteration=unittest.mock.Mock())
        callback = CheckpointCallback(
            save_dir=".",
            save_interval=999_999,
            metrics_logger=metrics_logger,
            verbose=0,
        )
        callback.locals = {
            "infos": [
                {"episode": {"r": 10, "l": 20, "max_x": 100,
                             "goal_reached": True}},
                {"episode": {"r": -5, "l": 10, "max_x": 40,
                             "goal_reached": False}},
            ]
        }
        callback.num_timesteps = 2048

        callback._on_step()

        logged = metrics_logger.log_iteration.call_args.args[0]
        self.assertEqual(logged["goal_rate"], 0.5)

    def test_atomic_live_model_records_matching_timestep_metadata(self):
        class FakeModel:
            @staticmethod
            def save(path):
                Path(path).write_bytes(b"checkpoint")

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = str(Path(temp_dir) / "model_live.zip")
            _save_model_atomic(FakeModel(), model_path, timestep=1234)

            metadata = json.loads(
                Path(_model_metadata_path(model_path)).read_text()
            )
            generation_path = Path(temp_dir) / metadata["model_file"]
            model_stat = generation_path.stat()
            self.assertEqual(metadata["timestep"], 1234)
            self.assertEqual(metadata["model_mtime_ns"], model_stat.st_mtime_ns)
            self.assertEqual(metadata["model_size"], model_stat.st_size)
            self.assertTrue(generation_path.is_file())
            self.assertFalse(Path(model_path).exists())

    def test_windows_replace_retries_a_temporary_reader_lock(self):
        with (
            patch(
                "server.agent.os.replace",
                side_effect=[PermissionError("locked"), None],
            ) as replace,
            patch("server.agent.time.sleep") as sleep,
        ):
            _replace_with_retry("source.zip", "target.zip")

        self.assertEqual(replace.call_count, 2)
        sleep.assert_called_once()

    def test_live_mirror_failure_does_not_stop_training(self):
        callback = CheckpointCallback(
            save_dir=".",
            save_interval=999_999,
            live_model_path="model_live.zip",
            live_save_interval=1,
            verbose=0,
        )
        callback.locals = {"infos": []}
        callback.num_timesteps = 1
        callback.model = object()

        with patch(
            "server.agent._save_model_atomic",
            side_effect=PermissionError("locked"),
        ):
            self.assertTrue(callback._on_step())


if __name__ == "__main__":
    unittest.main()
