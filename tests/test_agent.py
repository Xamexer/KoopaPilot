import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from server.agent import create_agent


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


if __name__ == "__main__":
    unittest.main()
