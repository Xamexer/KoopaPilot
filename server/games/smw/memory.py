"""Decode Super Mario World state from RetroJet's generic RAM capture.

This is intentionally the only place where the RetroJet training path knows
SMW RAM addresses and Map16/sprite semantics.  The decoder mirrors the former
native RetroJet decoder so existing observations, rewards, and PPO models keep
their exact schema.
"""

from __future__ import annotations

import math

import numpy as np

# Three contiguous reads cover all addresses used by the decoder while
# avoiding one Python/native call per field. Values are (region, start, len).
MEMORY_RANGES = (
    ("system_ram", 0x0000, 0x166E),
    ("system_ram", 0xC800, 0x4000),
    ("system_ram", 0x1C800, 0x3800),
)
CAPTURE_SIZE = sum(length for _, _, length in MEMORY_RANGES)


class PackedMemory:
    """Addressed view over one env-major, concatenated MemoryPlan result."""

    __slots__ = ("_data",)

    def __init__(self, data):
        view = memoryview(data)
        if view.ndim != 1 or view.format != "B":
            view = view.cast("B")
        if len(view) != CAPTURE_SIZE:
            raise ValueError(
                f"SMW RAM capture must contain {CAPTURE_SIZE} bytes, "
                f"got {len(view)}"
            )
        self._data = view

    @staticmethod
    def _offset(address: int) -> int:
        if 0 <= address < 0x166E:
            return address
        if 0xC800 <= address < 0x10800:
            return 0x166E + address - 0xC800
        if 0x1C800 <= address < 0x20000:
            return 0x566E + address - 0x1C800
        raise IndexError(f"SMW address not present in capture: 0x{address:X}")

    def u8(self, address: int) -> int:
        return self._data[self._offset(address)]

    def s8(self, address: int) -> int:
        value = self.u8(address)
        return value - 256 if value >= 128 else value

    def u16(self, address: int) -> int:
        return self.u8(address) | (self.u8(address + 1) << 8)


def split_captures(captured, num_envs: int) -> list[memoryview]:
    """Normalize RetroJet's env-major capture into one view per environment."""
    if isinstance(captured, (bytes, bytearray, memoryview)):
        flat = memoryview(captured)
        if flat.ndim != 1 or flat.format != "B":
            flat = flat.cast("B")
        expected = num_envs * CAPTURE_SIZE
        if len(flat) != expected:
            raise ValueError(
                f"expected {expected} captured bytes for {num_envs} envs, "
                f"got {len(flat)}"
            )
        return [
            flat[index * CAPTURE_SIZE:(index + 1) * CAPTURE_SIZE]
            for index in range(num_envs)
        ]

    rows = list(captured)
    if len(rows) != num_envs:
        raise ValueError(f"expected {num_envs} RAM captures, got {len(rows)}")
    return [memoryview(row).cast("B") for row in rows]


def decode_states(captured, num_envs: int, grid_size: int) -> list[dict]:
    """Decode an env-major capture with one vectorized model-compatible pass.

    The native session returns one packed ``bytes`` object.  NumPy views that
    buffer without copying it, then performs the address reads, Map16 lookup,
    and ordinary sprite-footprint lookup for every environment at once.  Only
    construction of the historical Python dict/list/bytes result is scalar.
    """
    if grid_size < 1 or grid_size % 2 == 0:
        raise ValueError("grid_size must be a positive odd number")

    data = _capture_matrix(captured, num_envs)
    mario_x_screen = np.minimum(0xFFFF, _batch_u16(data, 0x007E) + 8)
    mario_y_screen = np.minimum(0xFFFF, _batch_u16(data, 0x0080) + 8)
    mario_x_level = np.minimum(0xFFFF, _batch_u16(data, 0x00D1) + 8)
    mario_y_level = np.minimum(0xFFFF, _batch_u16(data, 0x00D3) + 8)
    camera_x = _batch_u16(data, 0x001A)
    camera_y = _batch_u16(data, 0x001C)
    screen_mode = _batch_u8(data, 0x005B)
    is_vertical = (screen_mode & 0x01) != 0

    tile_grids = _build_tile_grids_batch(
        data,
        mario_x_level,
        mario_y_level,
        camera_x,
        camera_y,
        is_vertical,
        grid_size,
    )
    sprites = _read_sprites_batch(data, camera_x, camera_y)

    powerup = _batch_u8(data, 0x0019)
    in_water = _batch_u8(data, 0x0075)
    in_air = _batch_u8(data, 0x0072)
    on_ground = _batch_u8(data, 0x13EF)
    climbing = _batch_u8(data, 0x0074)
    ducking = _batch_u8(data, 0x0073)
    player_anim = _batch_u8(data, 0x0071)
    coins = _batch_u8(data, 0x0DBF)
    lives = _batch_u8(data, 0x0DBE)
    game_mode = _batch_u8(data, 0x0100)
    sublevel = _batch_u8(data, 0x141A)
    num_screens = _batch_u8(data, 0x005D)
    goal_reached = _batch_u8(data, 0x1493)
    mario_x_speed = _batch_s8(data, 0x007B)
    mario_y_speed = _batch_s8(data, 0x007D)

    states = []
    for env_id in range(num_envs):
        states.append(
            {
                "type": "state",
                "mario_x_screen": int(mario_x_screen[env_id]),
                "mario_y_screen": int(mario_y_screen[env_id]),
                "mario_x_level": int(mario_x_level[env_id]),
                "mario_y_level": int(mario_y_level[env_id]),
                "camera_x": int(camera_x[env_id]),
                "camera_y": int(camera_y[env_id]),
                "powerup": int(powerup[env_id]),
                "in_water": int(in_water[env_id]),
                "in_air": int(in_air[env_id]),
                "on_ground": int(on_ground[env_id]),
                "climbing": int(climbing[env_id]),
                "ducking": int(ducking[env_id]),
                "player_anim": int(player_anim[env_id]),
                "coins": int(coins[env_id]),
                "lives": int(lives[env_id]),
                "game_mode": int(game_mode[env_id]),
                "sublevel": int(sublevel[env_id]),
                "num_screens": int(num_screens[env_id]),
                "screen_mode": int(screen_mode[env_id]),
                "goal_reached": int(goal_reached[env_id]),
                "mario_x_speed": int(mario_x_speed[env_id]),
                "mario_y_speed": int(mario_y_speed[env_id]),
                "is_vertical": bool(is_vertical[env_id]),
                "tile_grid": tile_grids[env_id],
                "sprites": sprites[env_id],
            }
        )
    return states


def _capture_matrix(captured, num_envs: int) -> np.ndarray:
    """Return a ``(env, byte)`` uint8 view, copying only row iterables."""
    expected = num_envs * CAPTURE_SIZE
    try:
        flat = memoryview(captured)
    except TypeError:
        rows = split_captures(captured, num_envs)
        matrix = np.empty((num_envs, CAPTURE_SIZE), dtype=np.uint8)
        for env_id, row in enumerate(rows):
            if len(row) != CAPTURE_SIZE:
                raise ValueError(
                    f"SMW RAM capture must contain {CAPTURE_SIZE} bytes, "
                    f"got {len(row)}"
                )
            matrix[env_id] = np.frombuffer(row, dtype=np.uint8)
        return matrix

    if flat.ndim != 1 or flat.format != "B":
        flat = flat.cast("B")
    if len(flat) != expected:
        raise ValueError(
            f"expected {expected} captured bytes for {num_envs} envs, "
            f"got {len(flat)}"
        )
    return np.frombuffer(flat, dtype=np.uint8).reshape(num_envs, CAPTURE_SIZE)


def _batch_u8(data: np.ndarray, address: int) -> np.ndarray:
    return data[:, PackedMemory._offset(address)]


def _batch_u16(data: np.ndarray, address: int) -> np.ndarray:
    offset = PackedMemory._offset(address)
    low = data[:, offset].astype(np.int32)
    high = data[:, offset + 1].astype(np.int32)
    return low | (high << 8)


def _batch_s8(data: np.ndarray, address: int) -> np.ndarray:
    values = _batch_u8(data, address).astype(np.int16)
    return np.where(values >= 128, values - 256, values)


def _batch_bytes(data: np.ndarray, address: int, length: int) -> np.ndarray:
    offset = PackedMemory._offset(address)
    return data[:, offset:offset + length]


def decode_state(captured, grid_size: int = 21) -> dict:
    """Port of RetroJet's former `smw::decode_state`."""
    if grid_size < 1 or grid_size % 2 == 0:
        raise ValueError("grid_size must be a positive odd number")

    ram = PackedMemory(captured)
    mario_x_screen = min(0xFFFF, ram.u16(0x007E) + 8)
    mario_y_screen = min(0xFFFF, ram.u16(0x0080) + 8)
    mario_x_level = min(0xFFFF, ram.u16(0x00D1) + 8)
    mario_y_level = min(0xFFFF, ram.u16(0x00D3) + 8)
    camera_x = ram.u16(0x001A)
    camera_y = ram.u16(0x001C)
    screen_mode = ram.u8(0x005B)
    is_vertical = bool(screen_mode & 0x01)

    return {
        "type": "state",
        "mario_x_screen": mario_x_screen,
        "mario_y_screen": mario_y_screen,
        "mario_x_level": mario_x_level,
        "mario_y_level": mario_y_level,
        "camera_x": camera_x,
        "camera_y": camera_y,
        "powerup": ram.u8(0x0019),
        "in_water": ram.u8(0x0075),
        "in_air": ram.u8(0x0072),
        "on_ground": ram.u8(0x13EF),
        "climbing": ram.u8(0x0074),
        "ducking": ram.u8(0x0073),
        "player_anim": ram.u8(0x0071),
        "coins": ram.u8(0x0DBF),
        "lives": ram.u8(0x0DBE),
        "game_mode": ram.u8(0x0100),
        "sublevel": ram.u8(0x141A),
        "num_screens": ram.u8(0x005D),
        "screen_mode": screen_mode,
        "goal_reached": ram.u8(0x1493),
        "mario_x_speed": ram.s8(0x007B),
        "mario_y_speed": ram.s8(0x007D),
        "is_vertical": is_vertical,
        "tile_grid": _build_tile_grid(
            ram,
            mario_x_level,
            mario_y_level,
            camera_x,
            camera_y,
            is_vertical,
            grid_size,
        ),
        "sprites": _read_sprites(ram, camera_x, camera_y),
    }


def _build_tile_grid(
    ram: PackedMemory,
    mario_x: int,
    mario_y: int,
    camera_x: int,
    camera_y: int,
    is_vertical: bool,
    grid_size: int,
) -> list[bytes]:
    half = grid_size // 2
    grid = []
    for dy in range(-half, half + 1):
        row = []
        for dx in range(-half, half + 1):
            px = mario_x + dx * 16
            py = mario_y + dy * 16
            if px < 0 or py < 0:
                row.append(0)
                continue

            # Match the BizHawk Lua viewport and the former Rust decoder.
            screen_x = px - camera_x - 8
            screen_y = py - camera_y - 8
            if not (0 <= screen_x < 256 and 0 <= screen_y < 224):
                row.append(0)
                continue

            tile = _get_map16_tile(ram, px, py, is_vertical)
            row.append(_classify_tile(tile))
        # PyO3 exposed each former Vec<u8> row as bytes; preserve that public
        # state schema as well as the numeric tile values.
        grid.append(bytes(row))
    return grid


def _build_tile_grids_batch(
    data: np.ndarray,
    mario_x: np.ndarray,
    mario_y: np.ndarray,
    camera_x: np.ndarray,
    camera_y: np.ndarray,
    is_vertical: np.ndarray,
    grid_size: int,
) -> list[list[bytes]]:
    """Build every viewport grid with vectorized coordinate and Map16 reads."""
    num_envs = data.shape[0]
    half = grid_size // 2
    delta = (np.arange(grid_size, dtype=np.int32) - half) * 16
    px = mario_x[:, None, None] + delta[None, None, :]
    py = mario_y[:, None, None] + delta[None, :, None]

    screen_x = px - camera_x[:, None, None] - 8
    screen_y = py - camera_y[:, None, None] - 8
    in_viewport = (
        (px >= 0)
        & (py >= 0)
        & (screen_x >= 0)
        & (screen_x < 256)
        & (screen_y >= 0)
        & (screen_y < 224)
    )

    tile_x = px // 16
    tile_y = py // 16

    horizontal_screen = tile_x // 16
    horizontal_index = (
        horizontal_screen * 0x1B0
        + tile_y * 16
        + tile_x % 16
    )

    vertical_screen = tile_y // 16
    vertical_local_y = tile_y % 16
    vertical_page = vertical_screen // 2
    vertical_index = (
        (vertical_page << 8)
        | ((vertical_screen & 1) << 7)
        | (vertical_local_y << 4)
        | (tile_x & 0x0F)
    )
    indices = np.where(
        is_vertical[:, None, None], vertical_index, horizontal_index
    )

    # Both Map16 planes begin at fixed locations inside the packed memory
    # capture.  The high plane ends at SNES WRAM offset 0x1FFFF, so indices
    # 0x3800..0x3FFF deliberately contribute a zero high byte.
    env_indices = np.arange(num_envs)[:, None, None]
    map16_valid = (indices >= 0) & (indices < 0x4000)
    low_indices = np.clip(indices, 0, 0x3FFF)
    low = data[
        env_indices,
        PackedMemory._offset(0xC800) + low_indices,
    ].astype(np.uint16)
    tiles = np.where(map16_valid, low, 0)

    high_valid = map16_valid & (indices < 0x3800)
    high_indices = np.clip(indices, 0, 0x37FF)
    high = data[
        env_indices,
        PackedMemory._offset(0x1C800) + high_indices,
    ].astype(np.uint16)
    tiles |= np.where(high_valid, high << 8, 0).astype(np.uint16)

    classes = _classify_tiles(tiles)
    classes[~in_viewport] = 0
    return [
        [classes[env_id, row].tobytes() for row in range(grid_size)]
        for env_id in range(num_envs)
    ]


def _get_map16_tile(
    ram: PackedMemory, level_x: int, level_y: int, is_vertical: bool
) -> int:
    tile_x = level_x // 16
    tile_y = level_y // 16
    if is_vertical:
        screen = tile_y // 16
        local_y = tile_y % 16
        page = screen // 2
        index = (page << 8) | ((screen & 1) << 7) | (local_y << 4) | (tile_x & 0x0F)
    else:
        screen = tile_x // 16
        local_x = tile_x % 16
        index = screen * 0x1B0 + tile_y * 16 + local_x
    if index >= 0x4000:
        return 0
    low = ram.u8(0xC800 + index)
    try:
        high = ram.u8(0x1C800 + index)
    except IndexError:
        # SNES WRAM ends at offset 0x1FFFF. The former native decoder treated
        # a high Map16 byte beyond that boundary as zero.
        high = 0
    return low | (high << 8)


def _classify_tile(tile: int) -> int:
    if 0x000 <= tile <= 0x003:
        return 1
    if 0x02A <= tile <= 0x02E:
        return 2
    if 0x006 <= tile <= 0x01C:
        return 3
    if tile == 0x038:
        return 4
    if (
        tile in (0x004, 0x005, 0x1FF, 0x12F)
        or 0x1D2 <= tile <= 0x1D7
        or 0x159 <= tile <= 0x15C
    ):
        return 5
    if 0x133 <= tile <= 0x13F:
        return 6
    if (
        tile in (0x021, 0x022, 0x029, 0x114)
        or 0x117 <= tile <= 0x11D
        or 0x11F <= tile <= 0x12B
    ):
        return 7
    if tile == 0x11E:
        return 8
    if tile == 0x12E:
        return 9
    if 0x100 <= tile <= 0x10C:
        return 10
    if (
        tile in (0x130, 0x132)
        or 0x140 <= tile <= 0x158
        or 0x14F <= tile <= 0x16D
        or 0x1C4 <= tile <= 0x1C9
    ):
        return 11
    if tile in (0x1B4, 0x1B5):
        return 12
    if tile in (0x01F, 0x020):
        return 13
    if (
        tile == 0x1B6
        or 0x16E <= tile <= 0x181
        or 0x196 <= tile <= 0x19F
        or 0x1AA <= tile <= 0x1AE
        or 0x1B8 <= tile <= 0x1B9
        or 0x1BC <= tile <= 0x1BD
        or 0x1C0 <= tile <= 0x1C1
        or 0x1CA <= tile <= 0x1CB
    ):
        return 14
    if (
        tile == 0x1B7
        or 0x182 <= tile <= 0x195
        or 0x1A0 <= tile <= 0x1A9
        or 0x1AF <= tile <= 0x1B3
        or 0x1BA <= tile <= 0x1BB
        or 0x1BE <= tile <= 0x1BF
        or 0x1C2 <= tile <= 0x1C3
        or 0x1CC <= tile <= 0x1CD
    ):
        return 15
    return 0


def _classify_tiles(tiles: np.ndarray) -> np.ndarray:
    """Vector equivalent of ``_classify_tile``, including its precedence."""
    conditions = (
        (tiles <= 0x003),
        ((tiles >= 0x02A) & (tiles <= 0x02E)),
        ((tiles >= 0x006) & (tiles <= 0x01C)),
        (tiles == 0x038),
        (
            (tiles == 0x004)
            | (tiles == 0x005)
            | (tiles == 0x1FF)
            | (tiles == 0x12F)
            | ((tiles >= 0x1D2) & (tiles <= 0x1D7))
            | ((tiles >= 0x159) & (tiles <= 0x15C))
        ),
        ((tiles >= 0x133) & (tiles <= 0x13F)),
        (
            (tiles == 0x021)
            | (tiles == 0x022)
            | (tiles == 0x029)
            | (tiles == 0x114)
            | ((tiles >= 0x117) & (tiles <= 0x11D))
            | ((tiles >= 0x11F) & (tiles <= 0x12B))
        ),
        (tiles == 0x11E),
        (tiles == 0x12E),
        ((tiles >= 0x100) & (tiles <= 0x10C)),
        (
            (tiles == 0x130)
            | (tiles == 0x132)
            | ((tiles >= 0x140) & (tiles <= 0x158))
            | ((tiles >= 0x14F) & (tiles <= 0x16D))
            | ((tiles >= 0x1C4) & (tiles <= 0x1C9))
        ),
        ((tiles == 0x1B4) | (tiles == 0x1B5)),
        ((tiles == 0x01F) | (tiles == 0x020)),
        (
            (tiles == 0x1B6)
            | ((tiles >= 0x16E) & (tiles <= 0x181))
            | ((tiles >= 0x196) & (tiles <= 0x19F))
            | ((tiles >= 0x1AA) & (tiles <= 0x1AE))
            | ((tiles >= 0x1B8) & (tiles <= 0x1B9))
            | ((tiles >= 0x1BC) & (tiles <= 0x1BD))
            | ((tiles >= 0x1C0) & (tiles <= 0x1C1))
            | ((tiles >= 0x1CA) & (tiles <= 0x1CB))
        ),
        (
            (tiles == 0x1B7)
            | ((tiles >= 0x182) & (tiles <= 0x195))
            | ((tiles >= 0x1A0) & (tiles <= 0x1A9))
            | ((tiles >= 0x1AF) & (tiles <= 0x1B3))
            | ((tiles >= 0x1BA) & (tiles <= 0x1BB))
            | ((tiles >= 0x1BE) & (tiles <= 0x1BF))
            | ((tiles >= 0x1C2) & (tiles <= 0x1C3))
            | ((tiles >= 0x1CC) & (tiles <= 0x1CD))
        ),
    )
    return np.select(conditions, range(1, 16), default=0).astype(np.uint8)


_SPRITE_CLIPPING_DISP_X = (
    0x02, 0x02, 0x10, 0x14, 0x00, 0x00, 0x01, 0x08, 0xF8, 0xFE, 0x03, 0x06,
    0x01, 0x00, 0x06, 0x02, 0x00, 0xE8, 0xFC, 0xFC, 0x04, 0x00, 0xFC, 0x02,
    0x02, 0x02, 0x02, 0x02, 0x00, 0x02, 0xE0, 0xF0, 0xFC, 0xFC, 0x00, 0xF8,
    0xF4, 0xF2, 0x00, 0xFC, 0xF2, 0xF0, 0x02, 0x00, 0xF8, 0x04, 0x02, 0x02,
    0x08, 0x00, 0x00, 0x00, 0xFC, 0x03, 0x08, 0x00, 0x08, 0x04, 0xF8, 0x00,
)
_SPRITE_CLIPPING_WIDTH = (
    0x0C, 0x0C, 0x10, 0x08, 0x30, 0x50, 0x0E, 0x28, 0x20, 0x14, 0x01, 0x03,
    0x0D, 0x0F, 0x14, 0x24, 0x0F, 0x40, 0x08, 0x08, 0x18, 0x0F, 0x18, 0x0C,
    0x0C, 0x0C, 0x0C, 0x0C, 0x0A, 0x1C, 0x30, 0x30, 0x08, 0x08, 0x10, 0x20,
    0x38, 0x3C, 0x20, 0x18, 0x1C, 0x20, 0x0C, 0x10, 0x10, 0x08, 0x1C, 0x1C,
    0x10, 0x30, 0x30, 0x40, 0x08, 0x12, 0x34, 0x0F, 0x20, 0x08, 0x20, 0x10,
)
_SPRITE_CLIPPING_DISP_Y = (
    0x03, 0x03, 0xFE, 0x08, 0xFE, 0xFE, 0x02, 0x08, 0xFE, 0x08, 0x07, 0x06,
    0xFE, 0xFC, 0x06, 0xFE, 0xFE, 0xE8, 0x10, 0x10, 0x02, 0xFE, 0xF4, 0x08,
    0x13, 0x23, 0x33, 0x43, 0x0A, 0xFD, 0xF8, 0xFC, 0xE8, 0x10, 0x00, 0xE8,
    0x20, 0x04, 0x58, 0xFC, 0xE8, 0xFC, 0xF8, 0x02, 0xF8, 0x04, 0xFE, 0xFE,
    0xF2, 0xFE, 0xFE, 0xFE, 0xFC, 0x00, 0x08, 0xF8, 0x10, 0x03, 0x10, 0x00,
)
_SPRITE_CLIPPING_HEIGHT = (
    0x0A, 0x15, 0x12, 0x08, 0x0E, 0x0E, 0x18, 0x30, 0x10, 0x1E, 0x02, 0x03,
    0x16, 0x10, 0x14, 0x12, 0x20, 0x40, 0x34, 0x74, 0x0C, 0x0E, 0x18, 0x45,
    0x3A, 0x2A, 0x1A, 0x0A, 0x30, 0x1B, 0x20, 0x12, 0x18, 0x18, 0x10, 0x20,
    0x38, 0x14, 0x08, 0x18, 0x28, 0x1B, 0x13, 0x4C, 0x10, 0x04, 0x22, 0x20,
    0x1C, 0x12, 0x12, 0x12, 0x08, 0x20, 0x2E, 0x14, 0x28, 0x0A, 0x10, 0x0D,
)

_SPRITE_CLIPPING_DISP_X_ARRAY = np.asarray(
    _SPRITE_CLIPPING_DISP_X, dtype=np.uint8
).view(np.int8).astype(np.int16)
_SPRITE_CLIPPING_DISP_Y_ARRAY = np.asarray(
    _SPRITE_CLIPPING_DISP_Y, dtype=np.uint8
).view(np.int8).astype(np.int16)
_SPRITE_CLIPPING_WIDTH_ARRAY = np.asarray(
    _SPRITE_CLIPPING_WIDTH, dtype=np.int16
)
_SPRITE_CLIPPING_HEIGHT_ARRAY = np.asarray(
    _SPRITE_CLIPPING_HEIGHT, dtype=np.int16
)


def _signed_byte(value: int) -> int:
    return value - 256 if value >= 128 else value


def _standard_sprite_footprint(clipping_index: int) -> tuple[int, int, int, int]:
    if clipping_index >= len(_SPRITE_CLIPPING_WIDTH):
        raise ValueError(
            f"unsupported SMW sprite clipping index 0x{clipping_index:02X}"
        )
    return (
        _signed_byte(_SPRITE_CLIPPING_DISP_X[clipping_index]),
        _signed_byte(_SPRITE_CLIPPING_DISP_Y[clipping_index]),
        _SPRITE_CLIPPING_WIDTH[clipping_index],
        _SPRITE_CLIPPING_HEIGHT[clipping_index],
    )


def _circle_offset(angle: int, radius: int) -> int:
    radians = (angle & 0x01FF) * (2.0 * math.pi / 512.0)
    sine = math.sin(radians)
    magnitude = min(256, math.floor(abs(sine) * 256.0 + 0.000001))
    offset = radius * magnitude // 256
    return -offset if sine < 0.0 else offset


def _sprite_footprint(
    ram: PackedMemory, slot: int, sprite_id: int
) -> tuple[int, int, int, int]:
    if sprite_id == 0x5F:
        angle = ((ram.u8(0x1528 + slot) & 0x01) << 8) | ram.u8(0x151C + slot)
        return (
            _circle_offset((angle + 0x80) & 0xFFFF, 0x50) - 0x68,
            _circle_offset(angle, 0x50) - 0x0C,
            0x40,
            0x13,
        )
    if sprite_id in (0x59, 0x5A):
        state = ram.u8(0x00C2 + slot)
        radius = ram.u8(0x151C + slot)
        if state & 0x02:
            return 0, -radius, 16, radius * 2 + 16
        return -radius, 0, radius * 2 + 16, 16
    return _standard_sprite_footprint(ram.u8(0x1662 + slot) & 0x3F)


def _read_sprites(
    ram: PackedMemory, camera_x: int, camera_y: int
) -> list[dict]:
    sprites = []
    for slot in range(12):
        status = ram.u8(0x14C8 + slot)
        sprite_id = ram.u8(0x009E + slot)
        sprite_world_x = (ram.u8(0x14E0 + slot) << 8) | ram.u8(0x00E4 + slot)
        sprite_world_y = (ram.u8(0x14D4 + slot) << 8) | ram.u8(0x00D8 + slot)
        speed_x = ram.s8(0x00B6 + slot)
        speed_y = ram.s8(0x00AA + slot)
        misc_state = ram.u8(0x00C2 + slot)
        offset_x, offset_y, width, height = _sprite_footprint(
            ram, slot, sprite_id
        )
        world_x = float(sprite_world_x + offset_x) + float(width) / 2.0
        world_y = float(sprite_world_y + offset_y) + float(height) / 2.0
        sprites.append(
            {
                "active": int(status >= 0x08),
                "id": sprite_id,
                "status": status,
                "screen_x": world_x - float(camera_x),
                "screen_y": world_y - float(camera_y),
                "world_x": world_x,
                "world_y": world_y,
                "speed_x": speed_x,
                "speed_y": speed_y,
                "misc_state": misc_state,
                "hitbox_width": width,
                "hitbox_height": height,
            }
        )
    return sprites


def _read_sprites_batch(
    data: np.ndarray, camera_x: np.ndarray, camera_y: np.ndarray
) -> list[list[dict]]:
    """Decode the twelve SMW sprite slots for every environment in one pass."""
    status = _batch_bytes(data, 0x14C8, 12)
    sprite_id = _batch_bytes(data, 0x009E, 12)
    world_x_raw = (
        _batch_bytes(data, 0x14E0, 12).astype(np.int32) << 8
    ) | _batch_bytes(data, 0x00E4, 12).astype(np.int32)
    world_y_raw = (
        _batch_bytes(data, 0x14D4, 12).astype(np.int32) << 8
    ) | _batch_bytes(data, 0x00D8, 12).astype(np.int32)
    speed_x_raw = _batch_bytes(data, 0x00B6, 12).astype(np.int16)
    speed_y_raw = _batch_bytes(data, 0x00AA, 12).astype(np.int16)
    speed_x = np.where(speed_x_raw >= 128, speed_x_raw - 256, speed_x_raw)
    speed_y = np.where(speed_y_raw >= 128, speed_y_raw - 256, speed_y_raw)
    misc_state = _batch_bytes(data, 0x00C2, 12)

    special_circle = sprite_id == 0x5F
    special_platform = (sprite_id == 0x59) | (sprite_id == 0x5A)
    special = special_circle | special_platform
    clipping = _batch_bytes(data, 0x1662, 12) & 0x3F

    invalid = (~special) & (clipping >= len(_SPRITE_CLIPPING_WIDTH))
    if np.any(invalid):
        # np.argwhere uses the same env-major/slot-major order as the former
        # nested scalar loops, preserving which malformed slot raises first.
        env_id, slot = np.argwhere(invalid)[0]
        invalid_index = int(clipping[env_id, slot])
        raise ValueError(
            f"unsupported SMW sprite clipping index 0x{invalid_index:02X}"
        )

    safe_clipping = np.minimum(clipping, len(_SPRITE_CLIPPING_WIDTH) - 1)
    offset_x = _SPRITE_CLIPPING_DISP_X_ARRAY[safe_clipping].astype(np.int32)
    offset_y = _SPRITE_CLIPPING_DISP_Y_ARRAY[safe_clipping].astype(np.int32)
    width = _SPRITE_CLIPPING_WIDTH_ARRAY[safe_clipping].astype(np.int32)
    height = _SPRITE_CLIPPING_HEIGHT_ARRAY[safe_clipping].astype(np.int32)

    radius = _batch_bytes(data, 0x151C, 12).astype(np.int32)
    platform_vertical = (misc_state & 0x02) != 0
    platform_offset_x = np.where(platform_vertical, 0, -radius)
    platform_offset_y = np.where(platform_vertical, -radius, 0)
    platform_width = np.where(platform_vertical, 16, radius * 2 + 16)
    platform_height = np.where(platform_vertical, radius * 2 + 16, 16)
    offset_x = np.where(special_platform, platform_offset_x, offset_x)
    offset_y = np.where(special_platform, platform_offset_y, offset_y)
    width = np.where(special_platform, platform_width, width)
    height = np.where(special_platform, platform_height, height)

    # Preserve the scalar libm calculation for the uncommon rotating-platform
    # footprint.  Its floor boundary must remain byte-for-byte compatible with
    # saved traces; at most the matching slots take this tiny Python loop.
    angle_low = _batch_bytes(data, 0x151C, 12)
    angle_high = _batch_bytes(data, 0x1528, 12)
    for env_id, slot in np.argwhere(special_circle):
        angle = (
            (int(angle_high[env_id, slot]) & 0x01) << 8
        ) | int(angle_low[env_id, slot])
        offset_x[env_id, slot] = (
            _circle_offset((angle + 0x80) & 0xFFFF, 0x50) - 0x68
        )
        offset_y[env_id, slot] = _circle_offset(angle, 0x50) - 0x0C
        width[env_id, slot] = 0x40
        height[env_id, slot] = 0x13

    world_x = (
        (world_x_raw + offset_x).astype(np.float64)
        + width.astype(np.float64) / 2.0
    )
    world_y = (
        (world_y_raw + offset_y).astype(np.float64)
        + height.astype(np.float64) / 2.0
    )
    screen_x = world_x - camera_x[:, None]
    screen_y = world_y - camera_y[:, None]
    active = status >= 0x08

    all_sprites = []
    for env_id in range(data.shape[0]):
        env_sprites = []
        for slot in range(12):
            env_sprites.append(
                {
                    "active": int(active[env_id, slot]),
                    "id": int(sprite_id[env_id, slot]),
                    "status": int(status[env_id, slot]),
                    "screen_x": float(screen_x[env_id, slot]),
                    "screen_y": float(screen_y[env_id, slot]),
                    "world_x": float(world_x[env_id, slot]),
                    "world_y": float(world_y[env_id, slot]),
                    "speed_x": int(speed_x[env_id, slot]),
                    "speed_y": int(speed_y[env_id, slot]),
                    "misc_state": int(misc_state[env_id, slot]),
                    "hitbox_width": int(width[env_id, slot]),
                    "hitbox_height": int(height[env_id, slot]),
                }
            )
        all_sprites.append(env_sprites)
    return all_sprites
