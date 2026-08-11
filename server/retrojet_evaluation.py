"""Deterministic RetroJet playback with live video and parity traces."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .environment import DISCRETE_ACTIONS
from .observation import build_observation

logger = logging.getLogger(__name__)

SNES_FPS = 60.0988
TRACE_STATE_FIELDS = (
    "mario_x_screen",
    "mario_y_screen",
    "mario_x_level",
    "mario_y_level",
    "camera_x",
    "camera_y",
    "mario_x_speed",
    "mario_y_speed",
    "powerup",
    "in_water",
    "in_air",
    "on_ground",
    "climbing",
    "ducking",
    "player_anim",
    "game_mode",
    "sublevel",
    "goal_reached",
    "lives",
)


def run_retrojet_evaluation(
    config: dict,
    model_path: str | None,
    episodes: int = 3,
    level_id: int | None = None,
    show_window: bool = True,
    realtime: bool = True,
    output_dir: str | None = None,
) -> list[dict]:
    """Play a fixed model in one Snes9x core and record evidence."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecFrameStack

    from .demo import _find_latest_model_path
    from .retrojet_backend import create_retrojet_vec_env

    diagnostic_config = copy.deepcopy(config)
    diagnostic_config["_mode"] = "retrojet-evaluation"
    diagnostic_config["_backend"] = "retrojet"
    backend = diagnostic_config.setdefault("backend", {})
    backend["type"] = "retrojet"
    backend.setdefault("retrojet", {})["num_envs"] = 1
    diagnostic_config.setdefault("performance", {})["retrojet_threads"] = 1
    if level_id is not None:
        diagnostic_config.setdefault("level_loading", {})["levels"] = [
            f"0x{level_id:03X}"
        ]

    model_path = model_path or _find_latest_model_path(
        diagnostic_config["paths"].get("model_dir", "./models")
    )
    if not model_path or not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path or '<latest>'}")

    base_env = create_retrojet_vec_env(diagnostic_config)
    base_env.runner.set_video_capture(0, True)
    vec_env = base_env
    frame_stack = int(diagnostic_config.get("ppo", {}).get("frame_stack", 1))
    if frame_stack > 1:
        vec_env = VecFrameStack(base_env, n_stack=frame_stack)

    try:
        model = PPO.load(model_path, env=vec_env)
        output_dir = _new_output_dir(
            output_dir
            or diagnostic_config["paths"].get("video_dir", "./videos")
        )
    except Exception:
        base_env.runner.set_video_capture(0, False)
        vec_env.close()
        raise
    summary = []
    stopped = False
    logger.info("RetroJet parity evaluation ready; startup frames were not recorded.")
    logger.info("Model: %s", model_path)
    logger.info("Artifacts: %s", output_dir)

    try:
        import cv2

        for episode in range(1, max(1, int(episodes)) + 1):
            obs = vec_env.reset()
            total_reward = 0.0
            steps = 0
            max_x = 0
            goal = False
            writer = None
            video_path = output_dir / f"episode_{episode:03d}.mp4"
            trace_path = output_dir / f"episode_{episode:03d}.jsonl"
            next_frame_at = time.perf_counter()

            try:
                with trace_path.open("w", encoding="utf-8") as trace_file:
                    while True:
                        policy_obs_hash = _observation_hash(obs)
                        policy_state = base_env.last_states[0] or {}
                        action, _ = model.predict(obs, deterministic=True)
                        action_index = int(np.asarray(action).reshape(-1)[0])
                        obs, rewards, dones, infos = vec_env.step(action)
                        reward = float(rewards[0])
                        done = bool(dones[0])
                        info = infos[0]
                        state = base_env.last_states[0] or {}

                        steps += 1
                        total_reward += reward
                        max_x = max(max_x, int(state.get("mario_x_level", 0)))
                        goal = goal or bool(info.get("goal_reached", False))
                        row = _trace_row(
                            episode=episode,
                            step=steps,
                            action_index=action_index,
                            reward=reward,
                            total_reward=total_reward,
                            done=done,
                            info=info,
                            config=diagnostic_config,
                            policy_state=policy_state,
                            state=state,
                            policy_obs_hash=policy_obs_hash,
                            next_obs_hash=_observation_hash(obs),
                        )
                        trace_file.write(json.dumps(row, sort_keys=True) + "\n")

                        frame = _read_frame(base_env.runner)
                        if frame is not None:
                            annotated = _annotate_frame(
                                cv2,
                                frame,
                                episode,
                                steps,
                                total_reward,
                                state,
                            )
                            if writer is None:
                                height, width = annotated.shape[:2]
                                writer = cv2.VideoWriter(
                                    str(video_path),
                                    cv2.VideoWriter_fourcc(*"mp4v"),
                                    SNES_FPS / base_env.runner.frame_skip,
                                    (width, height),
                                )
                                if not writer.isOpened():
                                    raise RuntimeError(
                                        f"Could not open video writer: {video_path}"
                                    )
                            writer.write(annotated)
                            if show_window:
                                display = cv2.resize(
                                    annotated,
                                    None,
                                    fx=3,
                                    fy=3,
                                    interpolation=cv2.INTER_NEAREST,
                                )
                                cv2.imshow("RetroJet parity evaluation", display)
                                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                                    stopped = True

                        if realtime:
                            next_frame_at += base_env.runner.frame_skip / SNES_FPS
                            delay = next_frame_at - time.perf_counter()
                            if delay > 0:
                                time.sleep(delay)

                        if done or stopped:
                            break
            finally:
                if writer is not None:
                    writer.release()
                else:
                    logger.warning(
                        "RetroJet episode %s produced no capturable video frames.",
                        episode,
                    )

            result = {
                "episode": episode,
                "reward": total_reward,
                "steps": steps,
                "max_x": max_x,
                "goal_reached": goal,
                "video": str(video_path) if video_path.is_file() else None,
                "trace": str(trace_path),
            }
            summary.append(result)
            logger.info(
                "RetroJet episode %s: goal=%s reward=%.1f steps=%s max_x=%s",
                episode,
                goal,
                total_reward,
                steps,
                max_x,
            )
            if stopped:
                break
    finally:
        base_env.runner.set_video_capture(0, False)
        vec_env.close()
        if show_window:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "model": str(Path(model_path).resolve()),
                "episodes": summary,
                "stopped_by_user": stopped,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("RetroJet parity summary: %s", summary_path)
    return summary


def _new_output_dir(video_dir: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = Path(video_dir) / f"retrojet-parity-{timestamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _read_frame(runner) -> np.ndarray | None:
    captured = runner.video_frame(0)
    if captured is None:
        return None
    width, height, rgb24 = captured
    rgb = np.frombuffer(rgb24, dtype=np.uint8)
    expected = int(width) * int(height) * 3
    if rgb.size != expected:
        raise RuntimeError(
            f"RetroJet returned {rgb.size} video bytes, expected {expected}"
        )
    return rgb.reshape((int(height), int(width), 3))[:, :, ::-1].copy()


def _annotate_frame(cv2, frame, episode, step, reward, state):
    annotated = frame.copy()
    lines = (
        f"ep {episode}  step {step}  reward {reward:.1f}",
        f"x {state.get('mario_x_level', 0)}  mode {state.get('game_mode', 0):02X}  "
        f"anim {state.get('player_anim', 0):02X}  goal {state.get('goal_reached', 0):02X}",
    )
    for index, text in enumerate(lines):
        y = 13 + index * 13
        cv2.putText(
            annotated,
            text,
            (4, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            text,
            (4, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated


def _observation_hash(observation) -> str:
    array = np.ascontiguousarray(np.asarray(observation), dtype=np.float32)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _observation_component_hashes(observation, config: dict) -> dict[str, str]:
    array = np.ascontiguousarray(np.asarray(observation), dtype=np.float32).reshape(-1)
    grid_size = int(config.get("normalization", {}).get("grid_size", 21))
    tiles_end = 13 + grid_size * grid_size
    return {
        "full": _observation_hash(array),
        "header": _observation_hash(array[:13]),
        "tiles": _observation_hash(array[13:tiles_end]),
        "sprites": _observation_hash(array[tiles_end:]),
    }


def _state_snapshot(state: dict) -> dict:
    snapshot = {field: state.get(field) for field in TRACE_STATE_FIELDS}
    snapshot["active_sprites"] = [
        {
            "slot": index,
            "id": sprite.get("id"),
            "status": sprite.get("status"),
            "x": sprite.get("world_x"),
            "y": sprite.get("world_y"),
        }
        for index, sprite in enumerate(state.get("sprites", []))
        if sprite.get("active")
    ]
    return snapshot


def _trace_row(
    *,
    episode,
    step,
    action_index,
    reward,
    total_reward,
    done,
    info,
    config,
    policy_state,
    state,
    policy_obs_hash,
    next_obs_hash,
):
    row = {
        "episode": episode,
        "step": step,
        "action": action_index,
        "buttons": DISCRETE_ACTIONS[action_index],
        "reward": reward,
        "total_reward": total_reward,
        "done": done,
        "reward_event": info.get("reward_event", ""),
        "policy_observation_sha256": policy_obs_hash,
        "next_observation_sha256": next_obs_hash,
        "policy_state": _state_snapshot(policy_state),
        "policy_state_observation_hashes": _observation_component_hashes(
            build_observation(policy_state, config), config
        ),
        "state_observation_hashes": _observation_component_hashes(
            build_observation(state, config), config
        ),
    }
    row.update({field: state.get(field) for field in TRACE_STATE_FIELDS})
    row["active_sprites"] = _state_snapshot(state)["active_sprites"]
    return row
