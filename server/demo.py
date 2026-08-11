"""Visible BizHawk demo playback for trained models."""

from __future__ import annotations

import copy
import io
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .emulator_manager import EmulatorManager
from .environment import SMWEnvironment
from .socket_server import SocketServer
from .vec_env import SMWVecEnv

logger = logging.getLogger(__name__)


def run_demo_mode(
    config: dict,
    model_path: Optional[str] = None,
    num_emulators: int = 1,
    episodes: Optional[int] = None,
    no_launch: bool = False,
    base_port: Optional[int] = None,
    reload_on_change: bool = False,
    wait_for_model: bool = False,
    stop_event: Optional[threading.Event] = None,
    episode_callback: Optional[Callable[[dict], None]] = None,
):
    """Run a visible BizHawk model demo."""
    stop_event = stop_event or threading.Event()
    demo_config = _demo_config(config, num_emulators, base_port)
    num_emulators = int(demo_config["emulator"]["num_instances"])
    envs = [
        SMWEnvironment(emulator_id=i, config=demo_config)
        for i in range(num_emulators)
    ]

    def on_state(emulator_id: int, state: dict) -> dict:
        if emulator_id < len(envs):
            return envs[emulator_id].on_state_received(state)
        return {"type": "action", "action": [0] * 7}

    server = SocketServer(
        demo_config["emulator"]["base_port"],
        num_emulators,
        on_state,
        demo_config,
    )
    server.start()

    emu_manager = None
    if not no_launch:
        emu_manager = EmulatorManager(demo_config)
        try:
            emu_manager.launch_all()
        except FileNotFoundError as exc:
            logger.error(str(exc))
            server.stop()
            return

    try:
        if not _wait_for_connections(server, timeout=120, stop_event=stop_event):
            logger.error("Demo emulator did not connect.")
            return

        if emu_manager:
            emu_manager.arrange_windows()

        vec_env = SMWVecEnv(envs)
        frame_stack = demo_config.get("ppo", {}).get("frame_stack", 1)
        if frame_stack > 1:
            logger.info("Loading Stable-Baselines frame stack for demo playback...")
            from stable_baselines3.common.vec_env import VecFrameStack

            vec_env = VecFrameStack(vec_env, n_stack=frame_stack)

        model_path = model_path or _find_latest_model_path(
            demo_config["paths"].get("model_dir", "./models")
        )
        manager = DemoManager(
            vec_env=vec_env,
            config=demo_config,
            model_path=model_path,
            reload_on_change=reload_on_change,
            wait_for_model=wait_for_model,
            stop_event=stop_event,
            episode_callback=episode_callback,
        )
        manager.run(episodes=episodes)
    finally:
        server.stop()
        if emu_manager:
            emu_manager.close_all()


class DemoManager:
    """Predict actions from a saved PPO model and show them in BizHawk."""

    def __init__(
        self,
        vec_env,
        config: dict,
        model_path: Optional[str],
        reload_on_change: bool = False,
        wait_for_model: bool = False,
        stop_event: Optional[threading.Event] = None,
        episode_callback: Optional[Callable[[dict], None]] = None,
    ):
        self.vec_env = vec_env
        self.config = config
        self.model_path = model_path
        self.reload_on_change = reload_on_change
        self.wait_for_model = wait_for_model
        self.stop_event = stop_event or threading.Event()
        self.model = None
        self.model_mtime_ns = None
        self.model_timestep = None
        self.loaded_model_path = None
        self.last_wait_log = 0.0
        self.episode_callback = episode_callback

    def run(self, episodes: Optional[int] = None):
        completed = 0
        obs = self.vec_env.reset()
        totals = np.zeros(self.vec_env.num_envs, dtype=np.float32)
        steps = np.zeros(self.vec_env.num_envs, dtype=np.int64)
        valid_evaluation = np.ones(self.vec_env.num_envs, dtype=bool)

        logger.info("Demo mode active. Press Ctrl+C to stop.")
        if episodes:
            logger.info("Demo will stop after %s completed episodes.", episodes)

        try:
            while not self.stop_event.is_set():
                # A live mirror may change at any time. Once an episode has
                # started, keep its policy fixed so its evaluation is a clean
                # checkpoint measurement rather than a mixture of two models.
                reload_allowed = self.model is None or bool(np.all(steps == 0))
                has_model = self._ensure_model(allow_reload=reload_allowed)
                if has_model:
                    action, _ = self.model.predict(obs, deterministic=True)
                else:
                    action = np.zeros(self.vec_env.num_envs, dtype=np.int64)
                    valid_evaluation[:] = False

                obs, rewards, dones, infos = self.vec_env.step(action)
                totals += rewards
                steps += 1

                for idx, done in enumerate(dones):
                    if not done:
                        continue
                    info = infos[idx]
                    episode = info.get("episode", {})
                    reward = episode.get("r", float(totals[idx]))
                    length = episode.get("l", int(steps[idx]))
                    max_x = episode.get("max_x", info.get("mario_x", 0))
                    goal = episode.get("goal_reached", info.get("goal_reached", False))
                    completed += 1
                    logger.debug(
                        "Demo env %s episode %s: reward=%.1f steps=%s max_x=%s goal=%s",
                        idx,
                        completed,
                        reward,
                        length,
                        max_x,
                        goal,
                    )
                    if self.episode_callback and valid_evaluation[idx]:
                        evaluation = {
                            "reward": float(reward),
                            "length": int(length),
                            "max_x": float(max_x),
                            "goal_reached": bool(goal),
                        }
                        if self.model_timestep is not None:
                            evaluation["checkpoint_timestep"] = int(
                                self.model_timestep
                            )
                        self.episode_callback(evaluation)
                    totals[idx] = 0.0
                    steps[idx] = 0
                    valid_evaluation[idx] = True

                if episodes is not None and completed >= episodes:
                    break
        except KeyboardInterrupt:
            logger.info("Demo stopped.")

    def _ensure_model(self, allow_reload: bool = True) -> bool:
        from .agent import LIVE_MODEL_IO_LOCK

        with LIVE_MODEL_IO_LOCK:
            return self._ensure_model_locked(allow_reload)

    def _ensure_model_locked(self, allow_reload: bool = True) -> bool:
        reference_path = self.model_path or _find_latest_model_path(
            self.config["paths"].get("model_dir", "./models")
        )
        if not reference_path:
            if self.wait_for_model:
                self._log_waiting("Waiting for demo model: <latest>")
                return False
            raise FileNotFoundError("Model not found: <latest>")

        try:
            model_path, metadata = _resolve_model_reference(reference_path)
        except Exception as exc:
            if self.wait_for_model:
                self._log_waiting("Demo model pointer is not ready yet: %s", exc)
                return self.model is not None
            raise

        if not os.path.exists(model_path):
            if self.wait_for_model:
                self._log_waiting("Waiting for demo model: %s", model_path)
                return self.model is not None
            raise FileNotFoundError(f"Model not found: {model_path}")

        model_stat = os.stat(model_path)
        mtime_ns = model_stat.st_mtime_ns
        if self.model is not None and (
            not self.reload_on_change
            or (
                self.loaded_model_path == os.path.abspath(model_path)
                and self.model_mtime_ns == mtime_ns
            )
        ):
            return True
        if self.model is not None and not allow_reload:
            return True

        try:
            _validate_model_metadata(metadata, model_stat)
            logger.debug("Loading demo model: %s", model_path)
            from stable_baselines3 import PPO

            # Deserialize from an in-memory snapshot. On Windows, loading
            # directly from the path can keep the ZIP locked briefly after
            # PPO.load() returns and race with the trainer's next replace.
            with open(model_path, "rb") as model_file:
                model_bytes = model_file.read()
            candidate_model = PPO.load(
                io.BytesIO(model_bytes), env=self.vec_env
            )
            completed_stat = os.stat(model_path)
            if (
                completed_stat.st_mtime_ns != model_stat.st_mtime_ns
                or completed_stat.st_size != model_stat.st_size
            ):
                raise RuntimeError("demo model changed while it was being loaded")
            self.model = candidate_model
            self.loaded_model_path = os.path.abspath(model_path)
            self.model_mtime_ns = mtime_ns
            checkpoint_timestep = metadata.get("timestep")
            if checkpoint_timestep is None:
                checkpoint_timestep = getattr(self.model, "num_timesteps", None)
            self.model_timestep = (
                int(checkpoint_timestep)
                if checkpoint_timestep is not None
                else None
            )
            return True
        except Exception as exc:
            if self.wait_for_model:
                if self.model is not None:
                    self._log_waiting(
                        "Demo model is still being written; keeping previous model: %s",
                        exc,
                    )
                    return True
                self._log_waiting("Demo model is not ready yet: %s", exc)
                return False
            raise

    def _log_waiting(self, message: str, *args):
        now = time.monotonic()
        if now - self.last_wait_log >= 5.0:
            logger.info(message, *args)
            self.last_wait_log = now


def _resolve_model_reference(reference_path: str) -> tuple[str, dict]:
    """Resolve a logical live mirror to its immutable generation file."""
    from .agent import _model_metadata_path

    metadata_path = _model_metadata_path(reference_path)
    if not os.path.isfile(metadata_path):
        return reference_path, {}
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    model_file = metadata.get("model_file")
    if not model_file:
        return reference_path, metadata
    if os.path.basename(model_file) != model_file:
        raise ValueError("demo model pointer contains an invalid filename")
    model_path = os.path.join(os.path.dirname(metadata_path), model_file)
    return model_path, metadata


def _validate_model_metadata(metadata: dict, model_stat):
    """Require sidecar metadata to match the selected complete generation."""
    if not metadata:
        return
    if (
        int(metadata.get("model_mtime_ns", -1)) != model_stat.st_mtime_ns
        or int(metadata.get("model_size", -1)) != model_stat.st_size
    ):
        raise RuntimeError("demo model metadata does not match the model file yet")


def _demo_config(config: dict, num_emulators: int, base_port: Optional[int]) -> dict:
    demo_config = copy.deepcopy(config)
    demo_config["_mode"] = "demo"
    emu = demo_config.setdefault("emulator", {})
    emu["num_instances"] = max(1, int(num_emulators))
    emu["speed_percent"] = 100
    if base_port is not None:
        emu["base_port"] = int(base_port)

    flags = demo_config.setdefault("flags", {})
    flags["visibility"] = True
    flags["reward_display"] = True
    flags["button_input_display"] = True
    return demo_config


def _wait_for_connections(
    server: SocketServer,
    timeout: float,
    stop_event: threading.Event,
) -> bool:
    deadline = time.monotonic() + timeout
    for emulator_id in range(server.num_instances):
        while not stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.error("Timeout waiting for demo emulator %s", emulator_id)
                return False
            if server._ready_events[emulator_id].wait(min(0.5, remaining)):
                break
        else:
            return False
    logger.info("All %s demo emulator(s) connected.", server.num_instances)
    return True


def _find_latest_model_path(model_dir: str) -> Optional[str]:
    path = Path(model_dir)
    if not path.is_dir():
        return None

    candidates = []
    preferred = [
        "model_best.zip",
        "model_final.zip",
        "model_interrupt.zip",
    ]
    for name in preferred:
        candidate = path / name
        if candidate.is_file():
            candidates.append(candidate)

    candidates.extend(path.glob("model_step_*.zip"))
    if not candidates:
        return None

    def sort_key(candidate: Path):
        match = re.search(r"model_step_(\d+)\.zip$", candidate.name)
        step = int(match.group(1)) if match else -1
        return (candidate.stat().st_mtime, step)

    return str(max(candidates, key=sort_key))
