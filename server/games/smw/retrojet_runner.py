"""SMW compatibility layer over RetroJet's SNES Session API."""

from __future__ import annotations

from pathlib import Path

from .actions import (
    INTRO_ADVANCE_MASK,
    RELEASE_ALL_MASK,
    action_mask,
)
from .memory import MEMORY_RANGES, decode_states

SYSTEM_RAM = "system_ram"


class SMWRetroJetRunner:
    """Present decoded SMW states while RetroJet only handles raw emulation.

    Keeping all Session calls in this class makes the boundary explicit: the
    native package sees button masks, memory ranges, bytes, and savestates;
    KoopaPilot owns every SMW address and state transition.
    """

    def __init__(
        self,
        session,
        memory_plan_type,
        config: dict,
        savestate_paths: list[str],
        level_ids: list[int],
    ):
        self._session = session
        self._plan = _make_memory_plan(memory_plan_type)
        self.grid_size = int(config.get("normalization", {}).get("grid_size", 21))
        self.num_envs = int(session.num_envs)
        self.frame_skip = int(session.frame_skip)
        self.num_threads = int(getattr(session, "num_threads", 0))

        _initialize_smw_environments(
            session,
            self.num_envs,
            savestate_paths,
            level_ids,
        )
        _set_reset_from_current(session)

    def step(self, actions) -> list[dict]:
        masks = [action_mask(index) for index in actions]
        captured = self._session.step_and_capture(masks, self._plan)
        return decode_states(captured, self.num_envs, self.grid_size)

    def states(self) -> list[dict]:
        captured = _capture(self._session, self._plan)
        return decode_states(captured, self.num_envs, self.grid_size)

    def reset(self, indices) -> list[dict]:
        indices = list(indices)
        self._session.reset(indices)
        captured = _capture(self._session, self._plan, indices)
        return decode_states(captured, len(indices), self.grid_size)

    def reset_all(self) -> list[dict]:
        self._session.reset_all()
        return self.states()

    def read_u8(self, env_id: int, address: int) -> int:
        return _read_u8(self._session, env_id, address)

    def read_u16(self, env_id: int, address: int) -> int:
        data = _read_memory(self._session, env_id, address, 2)
        return data[0] | (data[1] << 8)

    def read_ram(self, env_id: int, address: int, length: int) -> bytes:
        return _read_memory(self._session, env_id, address, length)

    def write_u8(self, env_id: int, address: int, value: int) -> None:
        _write_u8(self._session, env_id, address, value)

    def write_u16(self, env_id: int, address: int, value: int) -> None:
        _write_memory(
            self._session,
            env_id,
            address,
            bytes((value & 0xFF, (value >> 8) & 0xFF)),
        )

    def save_state(self, env_id: int) -> bytes:
        return bytes(self._session.save_state(env_id))

    def restore_state(self, env_id: int, state_bytes) -> None:
        self._session.restore_state(env_id, bytes(state_bytes))

    def set_video_capture(self, env_id: int, enabled: bool) -> None:
        self._session.set_video_capture(env_id, enabled)

    def video_frame(self, env_id: int):
        return self._session.video_frame(env_id)

    def close(self) -> None:
        session, self._session = self._session, None
        close = getattr(session, "close", None)
        if callable(close):
            close()


def _make_memory_plan(memory_plan_type):
    """Build the one native capture plan reused for every training step."""
    try:
        plan = memory_plan_type()
    except TypeError:
        return memory_plan_type(list(MEMORY_RANGES))
    for region, offset, length in MEMORY_RANGES:
        plan.add(region, offset, length)
    return plan


def _initialize_smw_environments(
    session,
    num_envs: int,
    savestate_paths: list[str],
    level_ids: list[int],
) -> None:
    """Reproduce the former native boot/savestate/level initialization."""
    if savestate_paths:
        state_cache = {
            path: Path(path).read_bytes()
            for path in dict.fromkeys(savestate_paths)
        }
        for env_id in range(num_envs):
            path = savestate_paths[env_id % len(savestate_paths)]
            session.restore_state(env_id, state_cache[path])
        return

    if not level_ids:
        return

    env_levels = {
        env_id: int(level_ids[env_id % len(level_ids)])
        for env_id in range(num_envs)
    }
    _skip_intro(session, list(env_levels))
    _warp_to_levels(session, env_levels)


def _skip_intro(session, env_ids: list[int], max_iterations: int = 3000) -> None:
    """Advance each env as Start+A, one frame, then release for three frames."""
    active = list(env_ids)
    for _ in range(max_iterations):
        active = [
            env_id
            for env_id in active
            if _read_u8(session, env_id, 0x0100) not in (0x0E, 0x14)
        ]
        if not active:
            return
        _run_frames(session, 1, [INTRO_ADVANCE_MASK] * len(active), active)
        _run_frames(session, 3, [RELEASE_ALL_MASK] * len(active), active)

    modes = ", ".join(
        f"env {env_id}=0x{_read_u8(session, env_id, 0x0100):02X}"
        for env_id in active
    )
    raise RuntimeError(f"SMW intro did not reach the overworld: {modes}")


def _warp_to_levels(session, env_levels: dict[int, int]) -> None:
    """Use SMW's full overworld loader and capture the exact reset boundary."""
    for env_id, level_id in env_levels.items():
        translevel, target_submap = _level_id_to_translevel(level_id)
        for address, value in (
            (0x0071, 0x00),
            (0x141A, 0x00),
            (0x1493, 0x00),
            (0x13C6, 0x00),
            (0x13CE, 0x00),
            (0x13D2, 0x00),
            (0x13BF, translevel),
            (0x1F11, target_submap),
            (0x0109, translevel),
            (0x0DAE, 0x0F),
            (0x0DAF, 0x01),
            (0x0DB0, 0x00),
            (0x0DB1, 0x02),
            (0x0100, 0x0F),
        ):
            _write_u8(session, env_id, address, value)

    active = list(env_levels)
    saw_full_load = {env_id: False for env_id in active}
    failures = {}
    for _ in range(600):
        if not active:
            break
        _run_frames(session, 1, [RELEASE_ALL_MASK] * len(active), active)
        next_active = []
        for env_id in active:
            mode = _read_u8(session, env_id, 0x0100)
            if 0x11 <= mode <= 0x13:
                saw_full_load[env_id] = True
            if mode == 0x14:
                _write_u8(session, env_id, 0x0109, 0x00)
                _write_u8(session, env_id, 0x0071, 0x00)
                if not saw_full_load[env_id]:
                    failures[env_id] = "entered level without full-load modes 0x11-0x13"
                continue
            if mode == 0x0E or mode <= 0x07:
                _write_u8(session, env_id, 0x0109, 0x00)
                failures[env_id] = f"loader returned to game mode 0x{mode:02X}"
                continue
            next_active.append(env_id)
        active = next_active

    for env_id in active:
        _write_u8(session, env_id, 0x0109, 0x00)
        failures[env_id] = "loader exceeded 600 frames"

    if failures:
        details = "; ".join(
            f"env {env_id}, level 0x{env_levels[env_id]:03X}: {reason}"
            for env_id, reason in sorted(failures.items())
        )
        raise RuntimeError(f"failed to full-load SMW level(s): {details}")


def _level_id_to_translevel(level_id: int) -> tuple[int, int]:
    if 0x001 <= level_id <= 0x024:
        return level_id, 0
    if 0x101 <= level_id <= 0x1DB:
        return level_id - 0x0DC, 1
    raise ValueError(f"SMW level ID cannot be entered: 0x{level_id:03X}")


def _capture(session, plan, indices=None):
    if indices is None:
        return session.capture(plan)
    return session.capture(plan, list(indices))


def _set_reset_from_current(session) -> None:
    for env_id in range(int(session.num_envs)):
        session.capture_reset_state(env_id)


def _run_frames(session, frames: int, input_masks: list[int], indices: list[int]) -> None:
    session.run_frames(frames, input_masks, indices)


def _read_memory(session, env_id: int, address: int, length: int) -> bytes:
    return bytes(session.read_memory(env_id, SYSTEM_RAM, address, length))


def _read_u8(session, env_id: int, address: int) -> int:
    return _read_memory(session, env_id, address, 1)[0]


def _write_memory(session, env_id: int, address: int, data: bytes) -> None:
    session.write_memory(env_id, SYSTEM_RAM, address, data)


def _write_u8(session, env_id: int, address: int, value: int) -> None:
    _write_memory(session, env_id, address, bytes((value & 0xFF,)))
