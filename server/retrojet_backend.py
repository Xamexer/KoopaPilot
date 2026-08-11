"""RetroJet backend for headless libretro training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from .environment import DISCRETE_ACTIONS
from .config import get_level_ids
from .observation import build_observation, get_observation_size
from .reward import EpisodeTracker, compute_reward, init_tracker_from_state

logger = logging.getLogger(__name__)


class RetroJetVecEnv(VecEnv):
    """Stable-Baselines3 VecEnv backed by RetroJet's native batch runner."""

    def __init__(self, config: dict, runner=None):
        self.config = config
        self.runner = runner or _create_runner(config)
        self.num_retro_envs = int(self.runner.num_envs)
        self.render_mode = None

        obs_size = get_observation_size(config)
        observation_space = spaces.Box(
            low=-0.5, high=1.5, shape=(obs_size,), dtype=np.float32
        )
        action_space = spaces.Discrete(len(DISCRETE_ACTIONS))
        super().__init__(self.num_retro_envs, observation_space, action_space)

        self.trackers = [EpisodeTracker() for _ in range(self.num_envs)]
        self.step_counts = [0 for _ in range(self.num_envs)]
        self._observations = np.zeros((self.num_envs, obs_size), dtype=np.float32)
        self._rewards = np.zeros(self.num_envs, dtype=np.float32)
        self._dones = np.zeros(self.num_envs, dtype=bool)
        self._infos = [{} for _ in range(self.num_envs)]
        self.last_states = [None for _ in range(self.num_envs)]
        self._actions: Optional[np.ndarray] = None

    def reset(self):
        states = self.runner.reset_all()
        for idx, state in enumerate(states):
            state = _prepare_state(state, 0)
            self.last_states[idx] = state
            self.trackers[idx] = EpisodeTracker()
            self.step_counts[idx] = 0
            init_tracker_from_state(self.trackers[idx], state, self.config)
            self._observations[idx] = build_observation(state, self.config)
        return self._observations.copy()

    def step_async(self, actions):
        self._actions = np.asarray(actions, dtype=np.int64)

    def step_wait(self):
        if self._actions is None:
            raise RuntimeError("step_async must be called before step_wait")

        raw_states = self.runner.step(self._actions.tolist())
        reset_indices = []

        for idx, state in enumerate(raw_states):
            self.step_counts[idx] += 1
            state = _prepare_state(state, self.step_counts[idx])
            self.last_states[idx] = state
            obs = build_observation(state, self.config)
            reward, event, done = compute_reward(
                state, self.trackers[idx], self.config
            )
            truncated = done and self.trackers[idx].truncated

            info = {
                "total_reward": self.trackers[idx].total_reward,
                "mario_x": state.get("mario_x_level", 0),
                "goal_reached": self.trackers[idx].goal_reached,
                "TimeLimit.truncated": truncated,
            }
            if event:
                info["reward_event"] = event
            if done:
                info["terminal_observation"] = obs
                info["episode"] = {
                    "r": self.trackers[idx].total_reward,
                    "l": self.step_counts[idx],
                    "max_x": self.trackers[idx].max_x,
                    "goal_reached": self.trackers[idx].goal_reached,
                    "truncated": truncated,
                }
                reset_indices.append(idx)

            self._observations[idx] = obs
            self._rewards[idx] = reward
            self._dones[idx] = done
            self._infos[idx] = info

        if reset_indices:
            reset_states = self.runner.reset(reset_indices)
            for idx, state in zip(reset_indices, reset_states):
                state = _prepare_state(state, 0)
                self.trackers[idx] = EpisodeTracker()
                self.step_counts[idx] = 0
                init_tracker_from_state(self.trackers[idx], state, self.config)
                self._observations[idx] = build_observation(state, self.config)

        self._actions = None
        return (
            self._observations.copy(),
            self._rewards.copy(),
            self._dones.copy(),
            self._infos.copy(),
        )

    def close(self):
        # Releasing the native Runner immediately unloads every copied
        # libretro DLL and frees its per-environment state.  Relying on cyclic
        # Python garbage collection can otherwise retain dozens of cores after
        # repeated training/benchmark runs.
        runner, self.runner = self.runner, None
        close = getattr(runner, "close", None)
        if callable(close):
            close()

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        target_envs = self._get_indices(indices)
        if method_name == "read_u8":
            return [
                self.runner.read_u8(i, *method_args, **method_kwargs)
                for i in target_envs
            ]
        raise AttributeError(f"RetroJetVecEnv has no env_method {method_name!r}")

    def get_attr(self, attr_name, indices=None):
        if attr_name == "runner":
            return [self.runner for _ in self._get_indices(indices)]
        return [getattr(self, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        for _ in self._get_indices(indices):
            setattr(self, attr_name, value)

    def seed(self, seed=None):
        return [None] * self.num_envs

    def _get_indices(self, indices):
        if indices is None:
            return list(range(self.num_envs))
        if isinstance(indices, int):
            return [indices]
        return list(indices)


def create_retrojet_vec_env(config: dict) -> RetroJetVecEnv:
    return RetroJetVecEnv(config)


def _create_runner(config: dict):
    try:
        import retrojet
    except ImportError as exc:
        raise RuntimeError(
            "RetroJet backend requested, but the retrojet package is not "
            "installed. Build the sibling ../RetroJet repository with "
            "`uv run maturin develop --release`, then install it into this "
            "environment."
        ) from exc

    retro_cfg = _retrojet_config(config)
    core_path = str(_default_core_path(config, retro_cfg))
    rom_path = str(_default_rom_path(config, retro_cfg))
    num_envs = int(
        retro_cfg.get(
            "num_envs",
            config.get("emulator", {}).get("num_instances", 1),
        )
    )
    frame_skip = int(config.get("emulator", {}).get(
        "frame_skip", retro_cfg.get("frame_skip", 4)
    ))
    grid_size = int(config.get("normalization", {}).get("grid_size", 21))
    boot_frames = int(retro_cfg.get("boot_frames", 300))
    num_threads = int(config.get("performance", {}).get(
        "retrojet_threads", 0
    ))
    level_loading = config.get("level_loading", {})
    savestate_paths = _savestate_paths(config, retro_cfg)
    if "levels" in level_loading:
        level_ids = (
            get_level_ids(config)
            if level_loading.get("mode") == "level_loading"
            else []
        )
    else:
        # Compatibility for configs created before shared level settings were
        # canonical. Validation emits a deprecation warning for this path.
        level_ids = [
            int(level_id, 16) if isinstance(level_id, str) else int(level_id)
            for level_id in retro_cfg.get("level_ids", [])
        ]

    _validate_retrojet_paths(core_path, rom_path, savestate_paths)

    logger.info(
        "Starting RetroJet backend: core=%s rom=%s envs=%s "
        "frame_skip=%s threads=%s savestates=%s levels=%s",
        core_path,
        rom_path,
        num_envs,
        frame_skip,
        num_threads or "auto",
        len(savestate_paths),
        len(level_ids),
    )
    try:
        return retrojet.Runner(
            core_path=core_path,
            rom_path=rom_path,
            num_envs=num_envs,
            frame_skip=frame_skip,
            savestate_paths=savestate_paths,
            grid_size=grid_size,
            boot_frames=boot_frames,
            level_ids=level_ids,
            num_threads=num_threads,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "RetroJet failed while initializing the native libretro runner. "
            "All configured core/ROM/savestate paths exist, so if the nested "
            "error is still 'failed to initialize env 0', the remaining fault "
            "is inside RetroJet or the selected libretro core rather than "
            "KoopaPilot's VecEnv/PPO code. "
            f"core={core_path!r}, rom={rom_path!r}, "
            f"savestates={len(savestate_paths)}, levels={len(level_ids)}, "
            f"envs={num_envs}. Original error: {exc}"
        ) from exc


def _savestate_paths(config: dict, retro_cfg: dict) -> list[str]:
    """Return canonical savestate paths, with legacy RetroJet fallback."""
    level_loading = config.get("level_loading", {})
    raw_paths = level_loading.get("savestate_files")
    if raw_paths is None:
        raw_paths = retro_cfg.get("savestate_paths", [])

    if isinstance(raw_paths, (str, Path)):
        raw_paths = [raw_paths]

    paths = []
    for path in raw_paths or []:
        if not isinstance(path, (str, Path)):
            raise TypeError(
                "Savestate paths must be strings or pathlib.Path objects; "
                f"got {type(path).__name__}: {path!r}"
            )
        paths.append(str(Path(path).expanduser().resolve()))
    return paths


def _validate_retrojet_paths(
    core_path: str, rom_path: str, savestate_paths: list[str]
) -> None:
    """Fail before native initialization when an input file is missing."""
    missing = []
    for label, path in (("core", core_path), ("ROM", rom_path)):
        if not Path(path).is_file():
            missing.append(f"{label}: {path}")

    for index, path in enumerate(savestate_paths):
        if not Path(path).is_file():
            missing.append(f"savestate[{index}]: {path}")

    if missing:
        raise FileNotFoundError(
            "RetroJet cannot start because required input files are missing:\n  - "
            + "\n  - ".join(missing)
        )


def _retrojet_config(config: dict) -> dict:
    backend = config.get("backend", {})
    if isinstance(backend, dict):
        return backend.get("retrojet", {}) or {}
    return {}


def _default_core_path(config: dict, retro_cfg: dict) -> Path:
    if retro_cfg.get("core_path"):
        return Path(retro_cfg["core_path"])
    root = Path(__file__).resolve().parents[2]
    preferred = root / "RetroJet" / "cores" / "snes9x2010_libretro.dll"
    if preferred.exists():
        return preferred
    return root / "RetroJet" / "cores" / "snes9x_libretro.dll"


def _default_rom_path(config: dict, retro_cfg: dict) -> Path:
    if config.get("paths", {}).get("rom"):
        return Path(config["paths"]["rom"])
    if retro_cfg.get("rom_path"):
        return Path(retro_cfg["rom_path"])
    root = Path(__file__).resolve().parents[2]
    return root / "RetroJet" / "roms" / "Super Mario World.sfc"


def _prepare_state(state: dict, step: int) -> dict:
    # RetroJet returns a fresh Python dict on every decode, so adding the
    # episode counter in place avoids cloning the whole state for every env.
    state["step"] = step
    return state
