import json
import unittest

from server.socket_server import EmulatorConnection


class SocketProtocolTests(unittest.TestCase):
    def setUp(self):
        self.connection = EmulatorConnection(None, ("local", 0), 0)

    def test_parses_newline_delimited_message(self):
        self.connection.buffer = '{"type":"state"}\n'

        self.assertEqual(
            self.connection._try_parse_buffer(),
            {"type": "state"},
        )

    def test_parses_length_prefixed_message(self):
        payload = json.dumps({"type": "handshake"}, separators=(",", ":"))
        self.connection.buffer = f"{len(payload)} {payload}"

        self.assertEqual(
            self.connection._try_parse_buffer(),
            {"type": "handshake"},
        )


if __name__ == "__main__":
    unittest.main()
