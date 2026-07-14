"""Configuration loader and validator."""

import json
import os
from pathlib import Path


def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from JSON file."""
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        config = json.load(f)

    _resolve_paths(config, path.parent)
    _validate(config)
    return config


def _resolve_paths(config: dict, base_dir: Path):
    """Resolve relative paths to absolute."""
    paths = config.get("paths", {})
    for key in ["bizhawk_exe", "rom", "savestate_dir", "model_dir",
                "log_dir", "video_dir", "lua_script"]:
        if key in paths and not os.path.isabs(paths[key]):
            paths[key] = str((base_dir / paths[key]).resolve())

    backend = config.get("backend", {})
    if isinstance(backend, dict):
        retrojet = backend.get("retrojet", {})
        if isinstance(retrojet, dict):
            for key in ["core_path", "rom_path"]:
                if key in retrojet and not os.path.isabs(retrojet[key]):
                    retrojet[key] = str((base_dir / retrojet[key]).resolve())
            if "savestate_paths" in retrojet:
                retrojet["savestate_paths"] = [
                    str((base_dir / path).resolve())
                    if not os.path.isabs(path)
                    else path
                    for path in retrojet["savestate_paths"]
                ]


def _validate(config: dict):
    """Basic validation of config values."""
    emu = config.get("emulator", {})
    backend = config.get("backend", {})
    retrojet = backend.get("retrojet", {}) if isinstance(backend, dict) else {}
    is_retrojet = isinstance(backend, dict) and backend.get("type") == "retrojet"
    num_instances = (
        retrojet.get("num_envs", emu.get("num_instances", 1))
        if is_retrojet and isinstance(retrojet, dict)
        else emu.get("num_instances", 1)
    )
    base_port = emu.get("base_port", 9000)
    frame_skip = emu.get("frame_skip", 4)
    if num_instances < 1:
        raise ValueError("num_instances must be >= 1")
    if base_port <= 1024 or base_port + num_instances - 1 > 65535:
        raise ValueError("emulator ports must stay within the range 1025-65535")
    if frame_skip < 1:
        raise ValueError("frame_skip must be >= 1")

    ppo = config.get("ppo", {})
    if ppo.get("n_steps", 2048) <= 0:
        raise ValueError("n_steps must be > 0")
    if ppo.get("batch_size", 64) <= 0:
        raise ValueError("batch_size must be > 0")
    rollout_size = ppo.get("n_steps", 2048) * num_instances
    batch_size = ppo.get("batch_size", 64)
    if batch_size > rollout_size:
        raise ValueError("batch_size must not exceed n_steps * num_instances")
    if rollout_size % batch_size != 0:
        raise ValueError("n_steps * num_instances must be divisible by batch_size")
    if not 0 < ppo.get("gamma", 0.99) <= 1:
        raise ValueError("gamma must be within (0, 1]")
    if not 0 <= ppo.get("gae_lambda", 0.95) <= 1:
        raise ValueError("gae_lambda must be within [0, 1]")
    if ppo.get("clip_range_schedule", "constant") not in {"constant", "linear"}:
        raise ValueError("clip_range_schedule must be 'constant' or 'linear'")
    if ppo.get("target_kl") is not None and ppo["target_kl"] <= 0:
        raise ValueError("target_kl must be > 0 when configured")

    norm = config.get("normalization", {})
    grid_size = norm.get("grid_size", 15)
    if grid_size < 1 or grid_size % 2 == 0:
        raise ValueError("grid_size must be a positive odd number")
    if norm.get("tile_categories_count", 16) < 2:
        raise ValueError("tile_categories_count must be >= 2")
    if norm.get("max_sprite_hitbox_dimension", 128) <= 0:
        raise ValueError("max_sprite_hitbox_dimension must be > 0")

    level_loading = config.get("level_loading", {})
    level_load_mode = level_loading.get("mode", "savestate")
    if level_load_mode not in {"savestate", "level_loading"}:
        raise ValueError("level_loading.mode must be 'savestate' or 'level_loading'")
    if level_load_mode == "level_loading":
        levels = get_level_ids(config)
        if not levels:
            raise ValueError("level_loading.levels must not be empty in level_loading mode")
        invalid_levels = [
            level_id for level_id in levels
            if not (0x001 <= level_id <= 0x024 or 0x101 <= level_id <= 0x1DB)
        ]
        if invalid_levels:
            formatted = ", ".join(f"0x{level_id:03X}" for level_id in invalid_levels)
            raise ValueError(
                "level_loading.levels contains IDs that cannot be entered through "
                f"SMW's full overworld loader: {formatted}. Lunar Magic level IDs "
                "are hexadecimal; quote them, for example \"0x105\"."
            )

    if isinstance(backend, dict):
        backend_type = backend.get("type", "bizhawk")
        if backend_type not in {"bizhawk", "retrojet"}:
            raise ValueError("backend.type must be 'bizhawk' or 'retrojet'")
        if backend_type == "retrojet" and isinstance(retrojet, dict):
            if retrojet.get("num_envs", num_instances) < 1:
                raise ValueError("backend.retrojet.num_envs must be >= 1")
            if retrojet.get("frame_skip", frame_skip) < 1:
                raise ValueError("backend.retrojet.frame_skip must be >= 1")


def get_savestate_files(config: dict) -> list:
    """Get list of savestate file paths.
    
    First checks level_loading.savestate_files config entries.
    If empty, scans the savestate directory for .State files.
    """
    # First, check if savestate_files are explicitly specified in config
    level_loading = config.get("level_loading", {})
    explicit_files = level_loading.get("savestate_files", [])
    
    if explicit_files:
        # Accept file names relative to savestate_dir as well as paths
        # relative to the configuration file's directory.
        savestate_dir = Path(config["paths"].get("savestate_dir", "./savestates"))
        result = []
        for f in explicit_files:
            path = Path(f)
            if not path.is_absolute():
                candidates = [savestate_dir / path, savestate_dir.parent / path]
                path = next((candidate for candidate in candidates
                             if candidate.exists()), candidates[0])
            if path.exists():
                result.append(str(path.resolve()).replace("\\", "/"))
            else:
                print(f"WARNING: Savestate not found: {path}")
        if result:
            return result
    
    # Fallback: scan savestate directory
    ss_dir = config["paths"].get("savestate_dir", "")
    if not ss_dir or not os.path.isdir(ss_dir):
        return []
    files = []
    for f in os.listdir(ss_dir):
        if f.endswith(".State") or f.endswith(".state"):
            files.append(os.path.join(ss_dir, f).replace("\\", "/"))
    return sorted(files)


def get_level_ids(config: dict) -> list:
    """Parse level ID strings (e.g., '0x105') to integers."""
    levels = config.get("level_loading", {}).get("levels", [])
    result = []
    for lv in levels:
        if isinstance(lv, str):
            result.append(int(lv, 16))
        else:
            result.append(int(lv))
    return result
