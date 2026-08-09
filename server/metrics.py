"""Training metrics logging to JSON for dashboard consumption."""

import copy
import json
import os
import logging
import threading
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricsLogger:
    """Logs training metrics to JSON files."""

    def __init__(self, log_dir: str, config: dict):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._lock = threading.RLock()

        self.run_id, self.run_dir = _create_run_directory(log_dir)

        self.metrics_file = os.path.join(self.run_dir, "metrics.json")
        self.config_file = os.path.join(self.run_dir, "config.json")

        # Keep the run reproducible without leaking machine-specific absolute
        # paths into a file that may be shared with dashboard screenshots.
        with open(self.config_file, "w") as f:
            json.dump(_portable_config_snapshot(config), f, indent=2)

        backend = config.get("_backend") or config.get("backend", {}).get(
            "type", "bizhawk"
        )
        emulator = config.get("emulator", {})
        retrojet = config.get("backend", {}).get("retrojet", {})
        num_envs = (
            retrojet.get("num_envs", emulator.get("num_instances", 1))
            if backend == "retrojet"
            else emulator.get("num_instances", 1)
        )

        self.data = {
            "run_id": self.run_id,
            "start_time": datetime.now().isoformat(),
            "config_summary": {
                "backend": backend,
                "action_selection": "stochastic (training)",
                "num_envs": num_envs,
                "learning_rate": config.get("ppo", {}).get("learning_rate", 0),
                "n_steps": config.get("ppo", {}).get("n_steps", 0),
                "batch_size": config.get("ppo", {}).get("batch_size", 0),
                "torch_threads": config.get("performance", {}).get(
                    "torch_threads"
                ),
                "retrojet_threads": config.get("performance", {}).get(
                    "retrojet_threads"
                ),
                "gamma": config.get("ppo", {}).get("gamma", 0),
                "frame_skip": emulator.get("frame_skip", 4),
                "reset_mode": config.get("level_loading", {}).get(
                    "mode", "savestate"
                ),
                "levels": config.get("level_loading", {}).get("levels", []),
            },
            "iterations": [],
            "evaluations": [],
        }
        self._save()

    def log_iteration(self, metrics: dict):
        """Log a training iteration's metrics."""
        with self._lock:
            entry = dict(metrics)
            entry["timestamp"] = datetime.now().isoformat()
            self.data["iterations"].append(entry)
            self._save_unlocked()

    def log_evaluation(self, metrics: dict):
        """Log one deterministic viewer episode alongside training rollouts."""
        with self._lock:
            entry = dict(metrics)
            if "timestep" not in entry:
                checkpoint_timestep = entry.get("checkpoint_timestep")
                if checkpoint_timestep is not None:
                    entry["timestep"] = int(checkpoint_timestep)
                else:
                    iterations = self.data["iterations"]
                    entry["timestep"] = (
                        iterations[-1]["timestep"] if iterations else 0
                    )
            entry["timestamp"] = datetime.now().isoformat()
            entry.setdefault("source", "bizhawk_live_demo")
            entry.setdefault("action_selection", "deterministic")
            self.data["evaluations"].append(entry)
            self._save_unlocked()

    def save(self):
        """Save metrics with end time."""
        with self._lock:
            self.data["end_time"] = datetime.now().isoformat()
            self._save_unlocked()

    def _save(self):
        """Write metrics to JSON file."""
        with self._lock:
            self._save_unlocked()

    def _save_unlocked(self):
        try:
            temporary_file = f"{self.metrics_file}.tmp"
            with open(temporary_file, "w") as f:
                json.dump(self.data, f, indent=2)
            os.replace(temporary_file, self.metrics_file)
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


def _portable_config_snapshot(config: dict) -> dict:
    """Return a config copy whose filesystem paths are relative and portable."""
    snapshot = copy.deepcopy(config)
    base_dir = Path(snapshot.pop("_config_dir", Path.cwd()))
    snapshot.pop("_screenshot_dir", None)

    paths = snapshot.get("paths", {})
    for key, value in list(paths.items()):
        if isinstance(value, str):
            paths[key] = _relative_path(value, base_dir)

    retrojet = snapshot.get("backend", {}).get("retrojet", {})
    for key in ("core_path", "rom_path"):
        if isinstance(retrojet.get(key), str):
            retrojet[key] = _relative_path(retrojet[key], base_dir)
    if isinstance(retrojet.get("savestate_paths"), list):
        retrojet["savestate_paths"] = [
            _relative_path(path, base_dir)
            for path in retrojet["savestate_paths"]
        ]

    live_demo = snapshot.get("live_demo", {})
    if isinstance(live_demo.get("model_path"), str):
        live_demo["model_path"] = _relative_path(
            live_demo["model_path"], base_dir
        )
    return snapshot


def _create_run_directory(log_dir: str) -> tuple[str, str]:
    """Atomically claim a unique run directory on coarse Windows clocks."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for collision_index in range(10_000):
        run_id = (
            timestamp
            if collision_index == 0
            else f"{timestamp}_{collision_index:04d}"
        )
        run_dir = os.path.join(log_dir, f"run_{run_id}")
        try:
            os.makedirs(run_dir, exist_ok=False)
            return run_id, run_dir
        except FileExistsError:
            continue
    raise RuntimeError("Could not allocate a unique metrics run directory")


def _relative_path(value: str, base_dir: Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()
    try:
        relative = Path(os.path.relpath(path, base_dir)).as_posix()
    except ValueError:
        # Different Windows drives cannot be represented by a relative path.
        return path.name
    if not relative.startswith("."):
        relative = f"./{relative}"
    return relative
