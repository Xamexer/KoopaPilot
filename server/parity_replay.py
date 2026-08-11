"""Replay a RetroJet action trace in BizHawk and report divergence."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from .demo import _demo_config, _wait_for_connections
from .emulator_manager import EmulatorManager
from .environment import SMWEnvironment
from .games.smw.actions import DISCRETE_ACTIONS
from .evaluation import compile_video
from .observation import build_observation
from .retrojet_evaluation import (
    TRACE_STATE_FIELDS,
    _observation_component_hashes,
    _observation_hash,
)
from .socket_server import SocketServer
from .vec_env import SMWVecEnv

logger = logging.getLogger(__name__)


def run_bizhawk_replay(
    config: dict,
    trace_path: str,
    level_id: int | None = None,
    no_launch: bool = False,
    output_dir: str | None = None,
) -> dict:
    """Replay saved action indices and compare BizHawk with RetroJet."""
    expected_rows = _load_trace(trace_path)
    replay_config = _demo_config(config, 1, None)
    replay_config["_mode"] = "demo"
    replay_config["_backend"] = "bizhawk"
    if level_id is not None:
        replay_config.setdefault("level_loading", {})["levels"] = [
            f"0x{level_id:03X}"
        ]

    artifact_dir = _new_output_dir(
        output_dir or replay_config["paths"].get("video_dir", "./videos")
    )
    frames_dir = artifact_dir / "frames"
    frames_dir.mkdir()
    replay_config["_screenshot_dir"] = str(frames_dir.resolve()).replace("\\", "/")

    env = SMWEnvironment(emulator_id=0, config=replay_config)

    def on_state(emulator_id: int, state: dict) -> dict:
        if emulator_id == 0:
            return env.on_state_received(state)
        return {"type": "action", "action": [0] * 7}

    base_port = int(replay_config["emulator"]["base_port"])
    server = SocketServer(base_port, 1, on_state, replay_config)
    server.start()
    emulator_manager = None
    vec_env = None

    try:
        if not no_launch:
            emulator_manager = EmulatorManager(replay_config)
            emulator_manager.launch_all()
        if not _wait_for_connections(server, timeout=120, stop_event=_NeverStop()):
            raise RuntimeError("BizHawk did not connect for parity replay")
        if emulator_manager:
            emulator_manager.arrange_windows()

        base_env = SMWVecEnv([env])
        vec_env = base_env
        frame_stack = int(replay_config.get("ppo", {}).get("frame_stack", 1))
        if frame_stack > 1:
            from stable_baselines3.common.vec_env import VecFrameStack

            vec_env = VecFrameStack(base_env, n_stack=frame_stack)

        obs = vec_env.reset()
        for startup_frame in frames_dir.glob("frame_*.png"):
            startup_frame.unlink()
        comparison_path = artifact_dir / "comparison.jsonl"
        initial_retrojet_hash = expected_rows[0].get(
            "policy_observation_sha256"
        )
        initial_bizhawk_hash = _observation_hash(obs)
        initial_observation_matches = (
            initial_bizhawk_hash == initial_retrojet_hash
        )
        expected_initial_state = expected_rows[0].get("policy_state", {})
        initial_state_differences = _state_differences(
            expected_initial_state, env.last_state or {}
        )
        expected_initial_components = expected_rows[0].get(
            "policy_state_observation_hashes", {}
        )
        actual_initial_components = _observation_component_hashes(
            build_observation(env.last_state or {}, replay_config), replay_config
        )
        initial_component_matches = _component_matches(
            expected_initial_components, actual_initial_components
        )
        first_observation_divergence = (
            None if initial_observation_matches else 0
        )
        first_unstacked_divergence = (
            None if initial_component_matches.get("full") else 0
        )
        first_state_divergence = None
        replayed_steps = 0
        actual_done = False
        actual_goal = False

        with comparison_path.open("w", encoding="utf-8") as comparison_file:
            for expected in expected_rows:
                step = int(expected["step"])
                action_index = int(expected["action"])
                obs, rewards, dones, infos = vec_env.step(
                    np.asarray([action_index], dtype=np.int64)
                )
                actual = env.last_transition_state or {}
                actual_hash = _observation_hash(obs)
                expected_hash = expected.get("next_observation_sha256")
                state_differences = _state_differences(expected, actual)
                observation_matches = actual_hash == expected_hash
                expected_components = expected.get(
                    "state_observation_hashes", {}
                )
                actual_components = _observation_component_hashes(
                    build_observation(actual, replay_config), replay_config
                )
                component_matches = _component_matches(
                    expected_components, actual_components
                )

                if not observation_matches and first_observation_divergence is None:
                    first_observation_divergence = step
                if state_differences and first_state_divergence is None:
                    first_state_divergence = step
                if (
                    not component_matches.get("full")
                    and first_unstacked_divergence is None
                ):
                    first_unstacked_divergence = step

                actual_done = bool(dones[0])
                actual_goal = actual_goal or bool(
                    infos[0].get("goal_reached", False)
                )
                row = {
                    "step": step,
                    "action": action_index,
                    "buttons": DISCRETE_ACTIONS[action_index],
                    "retrojet_observation_sha256": expected_hash,
                    "bizhawk_observation_sha256": actual_hash,
                    "observation_matches": observation_matches,
                    "retrojet_state_observation_hashes": expected_components,
                    "bizhawk_state_observation_hashes": actual_components,
                    "state_observation_component_matches": component_matches,
                    "retrojet_done": bool(expected.get("done", False)),
                    "bizhawk_done": actual_done,
                    "bizhawk_reward": float(rewards[0]),
                    "bizhawk_reward_event": infos[0].get("reward_event", ""),
                    "state_differences": state_differences,
                }
                comparison_file.write(json.dumps(row, sort_keys=True) + "\n")
                replayed_steps += 1
                if actual_done:
                    break

        summary = {
            "source_trace": str(Path(trace_path).resolve()),
            "expected_steps": len(expected_rows),
            "replayed_steps": replayed_steps,
            "bizhawk_done": actual_done,
            "bizhawk_goal_reached": actual_goal,
            "initial_retrojet_observation_sha256": initial_retrojet_hash,
            "initial_bizhawk_observation_sha256": initial_bizhawk_hash,
            "initial_observation_matches": initial_observation_matches,
            "initial_state_differences": initial_state_differences,
            "initial_retrojet_state_observation_hashes": expected_initial_components,
            "initial_bizhawk_state_observation_hashes": actual_initial_components,
            "initial_state_observation_component_matches": initial_component_matches,
            "first_observation_divergence_step": first_observation_divergence,
            "first_unstacked_observation_divergence_step": first_unstacked_divergence,
            "first_state_divergence_step": first_state_divergence,
            "comparison": str(comparison_path),
        }
    finally:
        if vec_env is not None:
            vec_env.close()
        server.stop()
        if emulator_manager:
            emulator_manager.close_all()

    video_path = artifact_dir / "bizhawk-replay.mp4"
    if compile_video(str(frames_dir), str(video_path)):
        summary["video"] = str(video_path)
    shutil.rmtree(frames_dir, ignore_errors=True)
    summary_path = artifact_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "BizHawk replay: goal=%s steps=%s first observation divergence=%s",
        summary["bizhawk_goal_reached"],
        replayed_steps,
        first_observation_divergence,
    )
    logger.info("BizHawk parity artifacts: %s", artifact_dir)
    return summary


def _load_trace(trace_path: str) -> list[dict]:
    path = Path(trace_path)
    if not path.is_file():
        raise FileNotFoundError(f"RetroJet trace not found: {trace_path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            action = int(row.get("action", -1))
            if not 0 <= action < len(DISCRETE_ACTIONS):
                raise ValueError(
                    f"Invalid action {action} on trace line {line_number}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"RetroJet trace is empty: {trace_path}")
    return rows


def _state_differences(expected: dict, actual: dict) -> dict:
    differences = {}
    for field in TRACE_STATE_FIELDS:
        if field not in expected or field not in actual:
            continue
        expected_value = expected.get(field)
        actual_value = actual.get(field)
        if expected_value != actual_value:
            differences[field] = {
                "retrojet": expected_value,
                "bizhawk": actual_value,
            }
    return differences


def _component_matches(expected: dict, actual: dict) -> dict[str, bool]:
    return {
        component: bool(expected.get(component))
        and expected.get(component) == actual.get(component)
        for component in ("full", "header", "tiles", "sprites")
    }


def _new_output_dir(parent: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = Path(parent) / f"bizhawk-parity-{timestamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


class _NeverStop:
    def is_set(self) -> bool:
        return False
