"""Multi-client TCP server for BizHawk emulator communication."""

import json
import socket
import threading
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class EmulatorConnection:
    """Handles communication with a single BizHawk emulator instance."""

    def __init__(self, conn: socket.socket, addr: tuple, emulator_id: int):
        self.conn = conn
        self.addr = addr
        self.emulator_id = emulator_id
        self.buffer = ""
        self.connected = True
        self.lock = threading.Lock()

    def receive_message(self) -> Optional[dict]:
        """Receive a message from BizHawk.

        BizHawk's comm.socketServerSend() may send in different formats:
        - Raw string with newline delimiter: "json_data\\n"
        - Length-prefixed: "LENGTH json_data" (no newline)
        We handle both cases.
        """
        try:
            # Read data until we have something to parse
            while True:
                # Try to parse what we have
                if self.buffer:
                    msg = self._try_parse_buffer()
                    if msg is not None:
                        return msg

                data = self.conn.recv(8192)
                if not data:
                    self.connected = False
                    return None
                self.buffer += data.decode("utf-8", errors="replace")

        except (ConnectionError, OSError):
            self.connected = False
            return None
        except json.JSONDecodeError as e:
            logger.warning(
                f"[Emu {self.emulator_id}] JSON decode error: {e} "
                f"| raw: {self.buffer[:200]!r}"
            )
            # Discard the bad line and continue
            if "\n" in self.buffer:
                _, self.buffer = self.buffer.split("\n", 1)
            else:
                self.buffer = ""
            return None

    def _try_parse_buffer(self) -> Optional[dict]:
        """Try to extract a JSON message from the buffer.

        Handles both newline-delimited and length-prefixed formats.
        """
        stripped = self.buffer.lstrip()

        # Format 1: Length-prefixed "LENGTH payload" (BizHawk socketServerSend)
        # The length number comes first, then a space, then the JSON payload
        import re
        length_match = re.match(r'^(\d+)\s', stripped)
        if length_match:
            length = int(length_match.group(1))
            prefix_len = length_match.end()
            # Check if we have enough data for the full payload
            if len(stripped) >= prefix_len + length:
                payload = stripped[prefix_len:prefix_len + length]
                # Advance buffer past this message
                consumed = len(self.buffer) - len(stripped) + prefix_len + length
                self.buffer = self.buffer[consumed:].lstrip('\n\r')
                try:
                    return json.loads(payload)
                except json.JSONDecodeError:
                    pass  # Fall through to newline-delimited parsing

        # Format 2: Newline-delimited JSON
        if "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            line = line.strip()
            if not line:
                return None  # Empty line, try again
            # Strip length prefix if present (e.g., "42 {...}")
            length_stripped = re.sub(r'^\d+\s+', '', line)
            if length_stripped:
                return json.loads(length_stripped)

        return None  # Need more data

    def send_response(self, response: dict):
        """Send a JSON response. BizHawk expects 'LENGTH payload' format."""
        try:
            payload = json.dumps(response, separators=(",", ":"))
            # BizHawk comm.socketServerResponse() expects: "LENGTH payload"
            message = f"{len(payload)} {payload}"
            with self.lock:
                self.conn.sendall(message.encode("utf-8"))
        except (ConnectionError, OSError):
            self.connected = False

    def close(self):
        """Close the connection."""
        self.connected = False
        try:
            self.conn.close()
        except OSError:
            pass


class SocketServer:
    """TCP server that accepts connections from multiple BizHawk instances."""

    def __init__(self, base_port: int, num_instances: int,
                 on_state: Callable, config: dict):
        self.base_port = base_port
        self.num_instances = num_instances
        self.on_state = on_state
        self.config = config
        self.connections: dict[int, EmulatorConnection] = {}
        self.servers: list[socket.socket] = []
        self.threads: list[threading.Thread] = []
        self.running = False
        self._ready_events: dict[int, threading.Event] = {}

    def start(self):
        """Start listening on all ports."""
        self.running = True
        for i in range(self.num_instances):
            port = self.base_port + i
            self._ready_events[i] = threading.Event()
            t = threading.Thread(
                target=self._listen_on_port,
                args=(port, i),
                daemon=True
            )
            t.start()
            self.threads.append(t)

        logger.info(
            f"Server listening on ports {self.base_port}-"
            f"{self.base_port + self.num_instances - 1}"
        )

    def wait_for_connections(self, timeout: float = 120.0) -> bool:
        """Wait for all emulators to connect."""
        for i in range(self.num_instances):
            if not self._ready_events[i].wait(timeout):
                logger.error(f"Timeout waiting for emulator {i} to connect")
                return False
        logger.info(f"All {self.num_instances} emulators connected.")
        return True

    def _listen_on_port(self, port: int, emulator_id: int):
        """Listen for a single emulator on a specific port."""
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.settimeout(1.0)
        try:
            server_sock.bind(("127.0.0.1", port))
        except OSError as e:
            logger.error(f"Cannot bind port {port}: {e}")
            return

        server_sock.listen(1)
        self.servers.append(server_sock)
        logger.info(f"Listening on port {port} for emulator {emulator_id}")

        while self.running:
            try:
                conn, addr = server_sock.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                logger.info(f"Emulator {emulator_id} connected from {addr}")

                emu_conn = EmulatorConnection(conn, addr, emulator_id)
                self.connections[emulator_id] = emu_conn
                self._ready_events[emulator_id].set()
                self._handle_connection(emu_conn)
            except socket.timeout:
                continue
            except OSError:
                if self.running:
                    logger.warning(f"Accept error on port {port}")
                break

    def _handle_connection(self, emu_conn: EmulatorConnection):
        """Handle messages from a connected emulator."""
        while self.running and emu_conn.connected:
            msg = emu_conn.receive_message()
            if msg is None:
                if not emu_conn.connected:
                    logger.info(f"Emulator {emu_conn.emulator_id} disconnected")
                    break
                continue

            msg_type = msg.get("type", "")

            if msg_type == "handshake":
                response = self._build_handshake_response(emu_conn.emulator_id)
                emu_conn.send_response(response)
            elif msg_type == "state":
                response = self.on_state(emu_conn.emulator_id, msg)
                emu_conn.send_response(response)
            else:
                emu_conn.send_response({"type": "ok"})

    def _build_handshake_response(self, emulator_id: int) -> dict:
        """Build config response for emulator handshake."""
        from .config import get_savestate_files, get_level_ids

        emu_cfg = self.config.get("emulator", {})
        flags = self.config.get("flags", {})
        ll = self.config.get("level_loading", {})

        return {
            "type": "config",
            "emulator_id": emulator_id,
            "frame_skip": emu_cfg.get("frame_skip", 4),
            "visibility": flags.get("visibility", True),
            "reward_display": flags.get("reward_display", True),
            "button_input_display": flags.get("button_input_display", True),
            "mode": self.config.get("_mode", "training"),
            "speed_percent": emu_cfg.get("speed_percent", 6400),
            "sound_enabled": emu_cfg.get("sound_enabled", False),
            "level_load_mode": ll.get("mode", "savestate"),
            "levels": get_level_ids(self.config),
            "savestate_files": get_savestate_files(self.config),
            "max_episode_steps": self.config.get("ppo", {}).get("max_episode_steps", 4500),
            "stagnation_timeout": self.config.get("ppo", {}).get("stagnation_timeout_steps", 600),
            "grid_size": self.config.get("normalization", {}).get("grid_size", 15),
            "screenshot_dir": self.config.get("_screenshot_dir"),
        }

    def send_to_emulator(self, emulator_id: int, message: dict):
        """Send a message to a specific emulator (outside normal flow)."""
        conn = self.connections.get(emulator_id)
        if conn and conn.connected:
            conn.send_response(message)

    def stop(self):
        """Stop the server and close all connections."""
        self.running = False
        for conn in self.connections.values():
            try:
                conn.send_response({"type": "close"})
            except Exception:
                pass
            conn.close()
        for s in self.servers:
            try:
                s.close()
            except OSError:
                pass
        self.connections.clear()
        logger.info("Server stopped.")
