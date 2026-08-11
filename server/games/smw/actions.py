"""KoopaPilot's Super Mario World action space.

RetroJet accepts raw Libretro joypad masks and deliberately has no knowledge
of this table.  The list representation remains available for the BizHawk Lua
protocol, so both backends continue to expose exactly the same 12 actions.
"""

# Libretro joypad bits used by the SNES cores.
BUTTON_B = 1 << 0
BUTTON_Y = 1 << 1
BUTTON_UP = 1 << 4
BUTTON_DOWN = 1 << 5
BUTTON_LEFT = 1 << 6
BUTTON_RIGHT = 1 << 7
BUTTON_A = 1 << 8

# Source layout: [Right, Left, Up, Down, A, B, ReleaseY].
# Y/run is held unless ReleaseY is set.
DISCRETE_ACTIONS = [
    [0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1],
]

# Exact masks formerly compiled into RetroJet. Keeping this explicit makes a
# model's action indices stable even if controller helpers change later.
DISCRETE_ACTION_MASKS = (
    0x002,
    0x082,
    0x083,
    0x182,
    0x042,
    0x043,
    0x142,
    0x003,
    0x102,
    0x022,
    0x012,
    0x000,
)

INTRO_ADVANCE_MASK = (1 << 3) | BUTTON_A  # Start + A, without implicit Y.
RELEASE_ALL_MASK = 0


def action_mask(action_index: int) -> int:
    """Translate a stable PPO action index to a raw Libretro joypad mask."""
    try:
        index = int(action_index)
    except (TypeError, ValueError):
        return DISCRETE_ACTION_MASKS[0]
    if not 0 <= index < len(DISCRETE_ACTION_MASKS):
        return DISCRETE_ACTION_MASKS[0]
    return DISCRETE_ACTION_MASKS[index]
