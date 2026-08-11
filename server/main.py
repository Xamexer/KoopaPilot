"""Main entry point for KoopaPilot."""

import argparse
import logging
import os
import threading
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("koopapilot")


def main():
    parser = argparse.ArgumentParser(description="KoopaPilot")
    parser.add_argument(
        "--mode",
        choices=[
            "training",
            "evaluation",
            "human",
            "dashboard",
            "demo",
            "live-demo",
            "retrojet-evaluation",
            "bizhawk-replay",
        ],
        default="training", help="Operation mode"
    )
    parser.add_argument("--config", default="config.json", help="Config file path")
    parser.add_argument("--model", default=None, help="Model path to load")
    parser.add_argument(
        "--backend",
        choices=["bizhawk", "retrojet"],
        default=None,
        help=(
            "Training backend override; live-demo always uses RetroJet and "
            "visible demo/evaluation/human modes use BizHawk"
        ),
    )
    parser.add_argument(
        "--no-launch", action="store_true",
        help="Don't launch emulators (connect to already running ones)"
    )
    parser.add_argument(
        "--dashboard-only", action="store_true",
        help="Only start the dashboard server"
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Number of evaluation or demo episodes"
    )
    parser.add_argument(
        "--demo-emulators", type=int, default=1,
        help="Number of visible BizHawk instances for demo mode"
    )
    parser.add_argument(
        "--live-demo-port", type=int, default=None,
        help="Base TCP port for the live-demo BizHawk viewer"
    )
    parser.add_argument(
        "--vis", action="store_true",
        help="Show colored tile-grid and sprite overlays in BizHawk"
    )
    parser.add_argument(
        "--level", type=lambda value: int(value, 0), default=None,
        help="Pin parity evaluation or replay to one level, for example 0x105"
    )
    parser.add_argument(
        "--no-window", action="store_true",
        help="Record RetroJet evaluation without opening a live video window"
    )
    parser.add_argument(
        "--no-realtime", action="store_true",
        help="Run RetroJet evaluation as fast as possible instead of real time"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Artifact directory override for parity evaluation or replay"
    )
    parser.add_argument(
        "--trace", default=None,
        help="RetroJet JSONL trace to replay in BizHawk"
    )
    parser.add_argument(
        "--lua-script", default=None,
        help="Lua script path override, useful when testing from a worktree"
    )
    args = parser.parse_args()

    # Load config
    from .config import load_config
    backend_override = _backend_override_for_mode(args.mode, args.backend)
    config = load_config(args.config, backend=backend_override)
    config["_mode"] = args.mode
    backend = backend_override or config.get("backend", {}).get(
        "type", "bizhawk"
    )
    config["_backend"] = backend
    if args.lua_script:
        config.setdefault("paths", {})["lua_script"] = os.path.abspath(
            args.lua_script
        )
    if args.vis:
        config.setdefault("flags", {})["visibility"] = True

    # Dashboard-only mode
    if args.dashboard_only or args.mode == "dashboard":
        from .dashboard.app import run_dashboard
        run_dashboard(config)
        return

    _configure_runtime(config)

    # The parity viewer writes its own artifacts and does not need a second
    # server thread or dashboard port during deterministic playback.
    if args.mode not in {"retrojet-evaluation", "bizhawk-replay"}:
        dash_thread = threading.Thread(
            target=_start_dashboard_background, args=(config,), daemon=True
        )
        dash_thread.start()

    if args.mode == "human":
        _run_human_mode(config, args)
    elif args.mode == "evaluation":
        _run_evaluation_mode(config, args)
    elif args.mode == "demo":
        _run_demo_mode(config, args)
    elif args.mode == "live-demo":
        _run_live_demo_mode(config, args)
    elif args.mode == "retrojet-evaluation":
        _run_retrojet_evaluation_mode(config, args)
    elif args.mode == "bizhawk-replay":
        _run_bizhawk_replay_mode(config, args)
    elif backend == "retrojet":
        _run_retrojet_training_mode(config, args)
    else:
        _run_training_mode(config, args)


def _start_dashboard_background(config):
    """Start dashboard in a background thread."""
    try:
        from .dashboard.app import create_app
        dash_cfg = config.get("dashboard", {})
        app = create_app(
            config["paths"].get("log_dir", "./logs"),
            dash_cfg.get("poll_interval_seconds", 5),
        )
        app.run(
            host=dash_cfg.get("host", "127.0.0.1"),
            port=dash_cfg.get("port", 8080),
            debug=False, use_reloader=False,
        )
    except Exception as e:
        logger.warning(f"Dashboard failed to start: {e}")


def _run_training_mode(config, args):
    """Run training mode."""
    from .socket_server import SocketServer
    from .environment import SMWEnvironment
    from .vec_env import SMWVecEnv
    from .emulator_manager import EmulatorManager
    from .training import TrainingManager

    num_envs = config["emulator"]["num_instances"]
    base_port = config["emulator"]["base_port"]

    # Create environments
    envs = []
    for i in range(num_envs):
        env = SMWEnvironment(emulator_id=i, config=config)
        envs.append(env)

    # State handler: routes state messages to the correct environment
    def on_state(emulator_id: int, state: dict) -> dict:
        if emulator_id < len(envs):
            return envs[emulator_id].on_state_received(state)
        return {"type": "action", "action": [0]*7}

    # Start socket server
    server = SocketServer(base_port, num_envs, on_state, config)
    server.start()
    logger.info("Socket server started. Waiting for emulators...")

    # Launch emulators
    emu_manager = None
    if not args.no_launch:
        emu_manager = EmulatorManager(config)
        try:
            emu_manager.launch_all()
        except FileNotFoundError as e:
            logger.error(str(e))
            server.stop()
            return

    # Wait for all connections
    if not server.wait_for_connections(timeout=120):
        logger.error("Not all emulators connected. Aborting.")
        if emu_manager:
            emu_manager.close_all()
        server.stop()
        return

    # Arrange windows
    if emu_manager:
        emu_manager.arrange_windows()

    # Create vectorized env and train
    vec_env = SMWVecEnv(envs)

    try:
        trainer = TrainingManager(vec_env, config, args.model)
        trainer.train()
    finally:
        logger.info("Shutting down...")
        server.stop()
        if emu_manager:
            emu_manager.close_all()


def _run_retrojet_training_mode(config, args, metrics_logger=None):
    """Run training mode through RetroJet instead of BizHawk."""
    logger.info("Loading RetroJet and PPO training stack...")
    from .retrojet_backend import create_retrojet_vec_env
    from .training import TrainingManager

    vec_env = create_retrojet_vec_env(config)
    try:
        trainer = TrainingManager(
            vec_env, config, args.model, metrics_logger=metrics_logger
        )
        trainer.train()
    finally:
        vec_env.close()


def _run_demo_mode(config, args):
    """Run visible model playback in BizHawk."""
    from .demo import run_demo_mode

    model_path = args.model or os.path.join(
        config["paths"].get("model_dir", "./models"), "model_best.zip"
    )
    run_demo_mode(
        config,
        model_path=model_path,
        num_emulators=args.demo_emulators,
        episodes=args.episodes,
        no_launch=args.no_launch,
    )


def _run_live_demo_mode(config, args):
    """Run RetroJet training while a visible BizHawk instance plays the model."""
    config["_backend"] = "retrojet"
    config.setdefault("backend", {})["type"] = "retrojet"

    model_dir = config["paths"].get("model_dir", "./models")
    model_path = _live_model_path(model_dir, args.model)
    from .agent import _cleanup_model_generations, _model_metadata_path

    for stale_path in (model_path, _model_metadata_path(model_path)):
        if os.path.exists(stale_path):
            os.remove(stale_path)
    _cleanup_model_generations(model_path, keep_latest=0)
    live_demo_cfg = config.setdefault("live_demo", {})
    live_demo_cfg["model_path"] = model_path
    live_demo_cfg.setdefault("save_interval_steps", 10_000)

    from .metrics import MetricsLogger

    log_dir = config["paths"].get("log_dir", "./logs")
    metrics_logger = MetricsLogger(log_dir, config)
    live_base_port = args.live_demo_port
    if live_base_port is None:
        live_base_port = int(config["emulator"].get("base_port", 9000)) + 1000

    stop_event = threading.Event()
    demo_thread = threading.Thread(
        target=_run_live_demo_thread,
        kwargs={
            "config": config,
            "model_path": model_path,
            "num_emulators": 1,
            "episodes": None,
            "no_launch": args.no_launch,
            "base_port": live_base_port,
            "reload_on_change": True,
            "wait_for_model": True,
            "stop_event": stop_event,
            "episode_callback": metrics_logger.log_evaluation,
        },
        daemon=True,
        name="live-demo",
    )

    logger.info(
        "Starting deterministic live viewer on port %s with mirror reference %s",
        live_base_port,
        model_path,
    )
    logger.info(
        "Dashboard training rewards use stochastic PPO rollouts; live viewer "
        "episodes are recorded separately as deterministic evaluations."
    )
    demo_thread.start()
    try:
        _run_retrojet_training_mode(
            config, args, metrics_logger=metrics_logger
        )
    finally:
        stop_event.set()
        demo_thread.join(timeout=5)


def _run_retrojet_evaluation_mode(config, args):
    """Show and record deterministic playback inside the Snes9x backend."""
    from .retrojet_evaluation import run_retrojet_evaluation

    run_retrojet_evaluation(
        config,
        model_path=args.model,
        episodes=args.episodes or 3,
        level_id=args.level,
        show_window=not args.no_window,
        realtime=not args.no_realtime,
        output_dir=args.output_dir,
    )


def _run_bizhawk_replay_mode(config, args):
    """Replay a recorded RetroJet action sequence in visible BizHawk."""
    if not args.trace:
        raise ValueError("--trace is required for --mode bizhawk-replay")
    from .parity_replay import run_bizhawk_replay

    run_bizhawk_replay(
        config,
        trace_path=args.trace,
        level_id=args.level,
        no_launch=args.no_launch,
        output_dir=args.output_dir,
    )


def _live_model_path(model_dir: str, resume_path: str | None) -> str:
    """Choose a writable viewer mirror that never aliases the resume source."""
    live_path = os.path.join(model_dir, "model_live.zip")
    if resume_path and _canonical_model_zip_path(resume_path) == (
        _canonical_model_zip_path(live_path)
    ):
        return os.path.join(model_dir, "model_live_viewer.zip")
    return live_path


def _canonical_model_zip_path(path: str) -> str:
    """Normalize an SB3 model path, including its implicit .zip suffix."""
    normalized = os.path.abspath(os.path.expanduser(path))
    if not normalized.lower().endswith(".zip"):
        normalized = f"{normalized}.zip"
    return os.path.normcase(os.path.realpath(normalized))


def _backend_override_for_mode(
    mode: str, requested_backend: str | None
) -> str | None:
    """Return the backend actually used by a mode before config validation."""
    if mode == "live-demo":
        return "retrojet"
    if mode == "retrojet-evaluation":
        return "retrojet"
    if mode == "bizhawk-replay":
        return "bizhawk"
    if mode in {"demo", "evaluation", "human"}:
        return "bizhawk"
    if mode == "training":
        return requested_backend
    return None


def _configure_runtime(config: dict):
    """Apply CPU settings before PPO creates its first tensor operation."""
    torch_threads = config.get("performance", {}).get("torch_threads")
    if torch_threads is None:
        return

    import torch

    torch.set_num_threads(int(torch_threads))
    logger.info("PyTorch CPU threads: %s", torch.get_num_threads())


def _run_live_demo_thread(**kwargs):
    """Import and run the visible BizHawk demo inside its own thread."""
    try:
        from .demo import run_demo_mode

        run_demo_mode(**kwargs)
    except Exception as exc:
        logger.error("Live demo stopped: %s", exc)


def _run_evaluation_mode(config, args):
    """Run evaluation mode."""
    from .socket_server import SocketServer
    from .environment import SMWEnvironment
    from .vec_env import SMWVecEnv
    from .emulator_manager import EmulatorManager
    from .evaluation import EvaluationManager

    # Evaluation uses 1 emulator at normal speed
    config["emulator"]["num_instances"] = 1
    config["emulator"]["speed_percent"] = 100
    config["flags"]["visibility"] = True
    config["flags"]["reward_display"] = True

    # Set up screenshot directory for video recording
    video_dir = config["paths"].get("video_dir", "./videos")
    screenshot_dir = os.path.join(video_dir, "frames")
    os.makedirs(screenshot_dir, exist_ok=True)
    config["_screenshot_dir"] = os.path.abspath(screenshot_dir).replace("\\", "/")

    envs = [SMWEnvironment(emulator_id=0, config=config)]

    def on_state(emulator_id: int, state: dict) -> dict:
        return envs[0].on_state_received(state)

    base_port = config["emulator"]["base_port"]
    server = SocketServer(base_port, 1, on_state, config)
    server.start()

    emu_manager = None
    if not args.no_launch:
        emu_manager = EmulatorManager(config)
        try:
            emu_manager.launch_all()
        except FileNotFoundError as e:
            logger.error(str(e))
            server.stop()
            return

    if not server.wait_for_connections(timeout=60):
        logger.error("Emulator did not connect.")
        if emu_manager:
            emu_manager.close_all()
        server.stop()
        return

    vec_env = SMWVecEnv(envs)
    frame_stack = config.get("ppo", {}).get("frame_stack", 1)
    if frame_stack > 1:
        from stable_baselines3.common.vec_env import VecFrameStack
        vec_env = VecFrameStack(vec_env, n_stack=frame_stack)
    model_path = args.model or os.path.join(
        config["paths"]["model_dir"], "model_best.zip"
    )

    try:
        num_savestates = len(config.get("level_loading", {}).get("savestate_files", []))
        num_episodes = args.episodes or max(num_savestates, 3)
        evaluator = EvaluationManager(vec_env, config, model_path)
        evaluator.evaluate(num_episodes=num_episodes)
    finally:
        server.stop()
        if emu_manager:
            emu_manager.close_all()


def _run_human_mode(config, args):
    """Run human play mode."""
    from .socket_server import SocketServer
    from .emulator_manager import EmulatorManager
    from .human_play import HumanPlayManager

    # Human mode: 1 emulator, normal speed, all overlays on
    config["emulator"]["num_instances"] = 1
    config["emulator"]["speed_percent"] = 100
    config["flags"]["visibility"] = True
    config["flags"]["reward_display"] = True
    config["flags"]["button_input_display"] = True

    human = HumanPlayManager(config)

    def on_state(emulator_id: int, state: dict) -> dict:
        return human.on_state(emulator_id, state)

    base_port = config["emulator"]["base_port"]
    server = SocketServer(base_port, 1, on_state, config)
    server.start()

    emu_manager = None
    if not args.no_launch:
        emu_manager = EmulatorManager(config)
        try:
            emu_manager.launch_all()
        except FileNotFoundError as e:
            logger.error(str(e))
            server.stop()
            return

    if not server.wait_for_connections(timeout=60):
        logger.error("Emulator did not connect.")
        if emu_manager:
            emu_manager.close_all()
        server.stop()
        return

    logger.info("Human play mode active. Press Ctrl+C to stop.")
    logger.info("Play the game normally - reward values are shown on the overlay.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        server.stop()
        if emu_manager:
            emu_manager.close_all()


if __name__ == "__main__":
    main()
