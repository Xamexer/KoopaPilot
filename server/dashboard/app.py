"""Flask dashboard for live training metrics visualization."""

import json
import os
import logging
from flask import Flask, render_template, jsonify

from ..metrics import MetricsLogger

logger = logging.getLogger(__name__)
logging.getLogger("werkzeug").setLevel(logging.WARNING)


def create_app(log_dir: str, poll_interval_seconds: float = 5) -> Flask:
    """Create the Flask dashboard application."""
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
    )
    @app.route("/")
    def index():
        return render_template(
            "index.html",
            poll_ms=max(1000, int(poll_interval_seconds * 1000)),
        )

    @app.route("/api/runs")
    def list_runs():
        """List all training runs."""
        runs = MetricsLogger.list_runs(log_dir)
        return jsonify(runs)

    @app.route("/api/metrics/<path:run_name>")
    def get_metrics(run_name):
        """Get metrics for a specific run."""
        if run_name != os.path.basename(run_name) or run_name in {".", ".."}:
            return jsonify({"error": "Invalid run name"}), 400
        metrics_path = os.path.join(log_dir, run_name, "metrics.json")
        if not os.path.isfile(metrics_path):
            return jsonify({"error": "Run not found"}), 404
        with open(metrics_path, "r") as f:
            data = json.load(f)
        return jsonify(data)

    @app.route("/api/latest")
    def get_latest():
        """Get the latest (most recent) run's metrics."""
        runs = MetricsLogger.list_runs(log_dir)
        if not runs:
            return jsonify({"error": "No runs found"}), 404
        latest = runs[-1]
        with open(latest["path"], "r") as f:
            data = json.load(f)
        return jsonify(data)

    return app


def run_dashboard(config: dict):
    """Start the dashboard server."""
    dash_cfg = config.get("dashboard", {})
    host = dash_cfg.get("host", "127.0.0.1")
    port = dash_cfg.get("port", 8080)
    log_dir = config["paths"].get("log_dir", "./logs")

    app = create_app(log_dir, dash_cfg.get("poll_interval_seconds", 5))
    logger.info(f"Dashboard running at http://{host}:{port}")
    app.run(host=host, port=port, debug=False, use_reloader=False)
