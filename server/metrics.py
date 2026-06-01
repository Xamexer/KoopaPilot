"""Training metrics logging to JSON for dashboard consumption."""

import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class MetricsLogger:
    """Logs training metrics to JSON files."""

    def __init__(self, log_dir: str, config: dict):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(log_dir, f"run_{self.run_id}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.metrics_file = os.path.join(self.run_dir, "metrics.json")
        self.config_file = os.path.join(self.run_dir, "config.json")

        # Save config snapshot
        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=2)

        self.data = {
            "run_id": self.run_id,
            "start_time": datetime.now().isoformat(),
            "config_summary": {
                "num_instances": config.get("emulator", {}).get("num_instances", 1),
                "learning_rate": config.get("ppo", {}).get("learning_rate", 0),
                "n_steps": config.get("ppo", {}).get("n_steps", 0),
                "batch_size": config.get("ppo", {}).get("batch_size", 0),
                "gamma": config.get("ppo", {}).get("gamma", 0),
                "frame_skip": config.get("emulator", {}).get("frame_skip", 4),
            },
            "iterations": [],
        }
        self._save()

    def log_iteration(self, metrics: dict):
        """Log a training iteration's metrics."""
        metrics["timestamp"] = datetime.now().isoformat()
        self.data["iterations"].append(metrics)
        self._save()

    def save(self):
        """Save metrics with end time."""
        self.data["end_time"] = datetime.now().isoformat()
        self._save()

    def _save(self):
        """Write metrics to JSON file."""
        try:
            with open(self.metrics_file, "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save metrics: {e}")

    @staticmethod
    def list_runs(log_dir: str) -> list:
        """List all available training runs."""
        runs = []
        if not os.path.isdir(log_dir):
            return runs
        for name in sorted(os.listdir(log_dir)):
            run_dir = os.path.join(log_dir, name)
            metrics_file = os.path.join(run_dir, "metrics.json")
            if os.path.isfile(metrics_file):
                runs.append({
                    "name": name,
                    "path": metrics_file,
                })
        return runs
