"""Observation building and normalization from raw game state."""

import numpy as np

# Default grid size, overridden by config
_GRID_SIZE = 15


def build_observation(state: dict, config: dict) -> np.ndarray:
    """Build normalized observation vector from game state.

    Layout (dynamic based on grid_size):
        [0-1]     Mario X/Y screen position (normalized 0-1)
        [2-3]     Mario X/Y speed (normalized -1 to 1)
        [4-7]     Powerup one-hot (small, big, cape, fire)
        [8-11]    State one-hot (ground, air, water, climbing)
        [12]      is_vertical (0 or 1)
        [13-...]  grid_size x grid_size tile grid (normalized 0-1)
        [...-end] 12 sprites x 9 values (active, id, x, y, speed_x, speed_y,
                  hitbox_width, hitbox_height, misc_state)
    """
    norm = config.get("normalization", {})
    screen_w = norm.get("screen_width", 256)
    screen_h = norm.get("screen_height", 224)
    tile_cats = norm.get("tile_categories_count", 16)
    max_hitbox_dimension = norm.get("max_sprite_hitbox_dimension", 128)
    grid_size = norm.get("grid_size", _GRID_SIZE)
    grid_cells = grid_size * grid_size
    header_size = 2 + 2 + 4 + 4 + 1  # pos + vel + powerup + state + is_vertical = 13

    obs = []

    # Mario screen position (normalized)
    mario_x = state.get("mario_x_screen", 128) / screen_w
    mario_y = state.get("mario_y_screen", 112) / screen_h
    obs.extend([
        np.clip(mario_x, 0.0, 1.0),
        np.clip(mario_y, 0.0, 1.0)
    ])

    # Mario velocity (signed, normalized to -1..1)
    mario_x_speed = state.get("mario_x_speed", 0) / 128.0
    mario_y_speed = state.get("mario_y_speed", 0) / 128.0
    obs.extend([
        np.clip(mario_x_speed, -1.0, 1.0),
        np.clip(mario_y_speed, -1.0, 1.0)
    ])

    # Powerup one-hot (4 categories: small=0, big=1, cape=2, fire=3)
    powerup = state.get("powerup", 0)
    powerup_oh = [0.0, 0.0, 0.0, 0.0]
    if 0 <= powerup <= 3:
        powerup_oh[powerup] = 1.0
    obs.extend(powerup_oh)

    # State one-hot (ground, air, water, climbing)
    on_ground = state.get("on_ground", 0) != 0
    in_air = state.get("in_air", 0) != 0
    in_water = state.get("in_water", 0) != 0
    climbing = state.get("climbing", 0) != 0

    if in_water:
        state_oh = [0.0, 0.0, 1.0, 0.0]
    elif climbing:
        state_oh = [0.0, 0.0, 0.0, 1.0]
    elif on_ground:
        state_oh = [1.0, 0.0, 0.0, 0.0]
    else:
        state_oh = [0.0, 1.0, 0.0, 0.0]  # in air by default
    obs.extend(state_oh)

    # Level type
    obs.append(1.0 if state.get("is_vertical", False) else 0.0)

    # Tile grid (grid_size x grid_size, normalized integer)
    tile_grid = state.get("tile_grid", [])
    grid_values = [
        cat
        for row in tile_grid[:grid_size]
        for cat in row[:grid_size]
    ][:grid_cells]
    for cat in grid_values:
        obs.append(cat / (tile_cats - 1))
    # Pad if grid is smaller than expected
    while len(obs) < header_size + grid_cells:
        obs.append(0.0)

    # Sprites: 12 slots x 9 values
    sprites = state.get("sprites", [])
    for i in range(12):
        if i < len(sprites):
            sp = sprites[i]
            active = float(sp.get("active", 0))
            if not active:
                obs.extend([0.0] * 9)
                continue
            sprite_id = sp.get("id", 0) / 255.0
            sx = np.clip(sp.get("screen_x", 128) / screen_w, -0.5, 1.5)
            sy = np.clip(sp.get("screen_y", 112) / screen_h, -0.5, 1.5)
            speed_x = np.clip(sp.get("speed_x", 0) / 128.0, -1.0, 1.0)
            speed_y = np.clip(sp.get("speed_y", 0) / 128.0, -1.0, 1.0)
            hitbox_width = np.clip(
                sp.get("hitbox_width", 16) / max_hitbox_dimension, 0.0, 1.0
            )
            hitbox_height = np.clip(
                sp.get("hitbox_height", 16) / max_hitbox_dimension, 0.0, 1.0
            )
            misc_state = sp.get("misc_state", 0) / 255.0
            obs.extend([
                active, sprite_id, sx, sy, speed_x, speed_y,
                hitbox_width, hitbox_height, misc_state,
            ])
        else:
            obs.extend([0.0] * 9)

    return np.array(obs, dtype=np.float32)


def get_observation_size(config: dict = None) -> int:
    """Return the total observation vector size."""
    if config:
        grid_size = config.get("normalization", {}).get("grid_size", _GRID_SIZE)
    else:
        grid_size = _GRID_SIZE
    return 2 + 2 + 4 + 4 + 1 + (grid_size * grid_size) + 108
