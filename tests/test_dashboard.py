import json
import tempfile
import unittest
from pathlib import Path

from server.dashboard.app import create_app


class DashboardTests(unittest.TestCase):
    def test_index_exposes_configured_poll_interval(self):
        client = create_app(".", poll_interval_seconds=2.5).test_client()

        response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"SMW_DASHBOARD_POLL_MS = 2500", response.data)
        self.assertIn(b'id="smoothingSlider"', response.data)

    def test_latest_and_named_run_endpoints(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run_20260101_120000"
            run_dir.mkdir()
            metrics = {"run_id": "20260101_120000", "iterations": []}
            (run_dir / "metrics.json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )

            client = create_app(temp_dir).test_client()

            self.assertEqual(client.get("/api/latest").get_json(), metrics)
            self.assertEqual(
                client.get("/api/metrics/run_20260101_120000").get_json(),
                metrics,
            )


if __name__ == "__main__":
    unittest.main()
