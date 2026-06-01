"""Launch, arrange, and manage BizHawk emulator instances."""

import logging
import os
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _find_windows_by_pid(pid: int) -> list:
    """Find all window handles belonging to a process."""
    try:
        import win32gui
        import win32process

        handles = []

        def callback(hwnd, _):
            _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
            if found_pid == pid and win32gui.IsWindowVisible(hwnd):
                handles.append(hwnd)

        win32gui.EnumWindows(callback, None)
        return handles
    except ImportError:
        logger.warning("pywin32 not installed, cannot arrange windows")
        return []


def _get_window_title(hwnd) -> str:
    try:
        import win32gui
        return win32gui.GetWindowText(hwnd)
    except Exception:
        return ""


def _move_window(hwnd, x: int, y: int, w: int, h: int):
    try:
        import win32gui
        win32gui.MoveWindow(hwnd, x, y, w, h, True)
    except Exception as e:
        logger.warning(f"Failed to move window: {e}")


def _minimize_window(hwnd):
    try:
        import win32gui
        import win32con
        win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
    except Exception:
        pass


class EmulatorManager:
    """Manages launching and arranging BizHawk emulator instances."""

    def __init__(self, config: dict):
        self.config = config
        self.processes: list[subprocess.Popen] = []
        self.num_instances = config.get("emulator", {}).get("num_instances", 1)
        self.base_port = config.get("emulator", {}).get("base_port", 9000)

    def launch_all(self):
        """Launch all BizHawk emulator instances."""
        bizhawk_exe = self.config["paths"]["bizhawk_exe"]
        rom_path = self.config["paths"]["rom"]
        lua_script = self.config["paths"]["lua_script"]

        if not os.path.exists(bizhawk_exe):
            raise FileNotFoundError(f"BizHawk not found: {bizhawk_exe}")
        if not os.path.exists(rom_path):
            raise FileNotFoundError(f"ROM not found: {rom_path}")
        if not os.path.exists(lua_script):
            raise FileNotFoundError(f"Lua script not found: {lua_script}")

        bizhawk_dir = str(Path(bizhawk_exe).parent)

        for i in range(self.num_instances):
            port = self.base_port + i
            cmd = [
                bizhawk_exe,
                f"--socket_ip=127.0.0.1",
                f"--socket_port={port}",
                f"--lua={os.path.abspath(lua_script)}",
                os.path.abspath(rom_path)
            ]
            logger.info(f"Launching emulator {i} on port {port}")
            proc = subprocess.Popen(
                cmd,
                cwd=bizhawk_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.processes.append(proc)
            time.sleep(0.5)  # Stagger launches

        logger.info(f"Launched {self.num_instances} emulator instances")

    def arrange_windows(self):
        """Arrange emulator windows in a grid and minimize Lua consoles."""
        emu_cfg = self.config.get("emulator", {})
        grid_cols = emu_cfg.get("grid_cols", 4)
        grid_rows = emu_cfg.get("grid_rows", 2)
        win_w = emu_cfg.get("window_width", 292)
        win_h = emu_cfg.get("window_height", 253)

        time.sleep(3.0)  # Wait for windows to appear

        for i, proc in enumerate(self.processes):
            if proc.poll() is not None:
                continue

            handles = _find_windows_by_pid(proc.pid)

            col = i % grid_cols
            row = i // grid_cols
            x = col * win_w
            y = row * win_h

            for hwnd in handles:
                title = _get_window_title(hwnd)
                if "Lua Console" in title or "LuaConsole" in title:
                    _minimize_window(hwnd)
                elif title:  # Emulator window
                    _move_window(hwnd, x, y, win_w, win_h)

        logger.info(
            f"Arranged windows in {grid_rows}x{grid_cols} grid"
        )

    def close_all(self):
        """Terminate all emulator processes."""
        for proc in self.processes:
            try:
                proc.terminate()
            except Exception:
                pass

        # Wait briefly then force kill
        time.sleep(1.0)
        for proc in self.processes:
            try:
                if proc.poll() is None:
                    proc.kill()
            except Exception:
                pass

        self.processes.clear()
        logger.info("All emulators closed.")

    def is_any_alive(self) -> bool:
        """Check if any emulator is still running."""
        return any(p.poll() is None for p in self.processes)
