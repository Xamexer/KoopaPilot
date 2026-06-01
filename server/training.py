"""Training loop with graceful shutdown and model saving."""

import logging
import os
import signal
import sys

from stable_baselines3.common.vec_env import VecFrameStack
from .agent import create_agent, CheckpointCallback
from .metrics import MetricsLogger

logger = logging.getLogger(__name__)


class TrainingManager:
    """Manages the PPO training loop with interrupt handling."""

    def __init__(self, vec_env, config: dict, model_path: str = None):
        # Apply frame stacking if configured
        frame_stack = config.get("ppo", {}).get("frame_stack", 1)
        if frame_stack > 1:
            vec_env = VecFrameStack(vec_env, n_stack=frame_stack)
            logger.info(f"Applied frame stacking: {frame_stack} frames")

        self.vec_env = vec_env
        self.config = config
        self.model_path = model_path
        self.model = None
        self.interrupted = False

        self.save_dir = config["paths"].get("model_dir", "./models")
        os.makedirs(self.save_dir, exist_ok=True)

    def train(self):
        """Run the training loop."""
        ppo_cfg = self.config.get("ppo", {})
        total_timesteps = ppo_cfg.get("total_timesteps", 10_000_000)
        save_interval = ppo_cfg.get("save_interval_steps", 50_000)

        # Setup metrics
        log_dir = self.config["paths"].get("log_dir", "./logs")
        os.makedirs(log_dir, exist_ok=True)
        metrics_logger = MetricsLogger(log_dir, self.config)

        # Create agent
        self.model = create_agent(self.vec_env, self.config, self.model_path)

        # Setup callback
        callback = CheckpointCallback(
            save_dir=self.save_dir,
            save_interval=save_interval,
            metrics_logger=metrics_logger,
        )

        # Setup interrupt handler
        original_sigint = signal.getsignal(signal.SIGINT)

        def _handle_interrupt(signum, frame):
            logger.info("\nInterrupt received! Saving model...")
            self.interrupted = True
            self._save_interrupt()
            signal.signal(signal.SIGINT, original_sigint)
            sys.exit(0)

        signal.signal(signal.SIGINT, _handle_interrupt)

        logger.info(f"Starting training for {total_timesteps} timesteps")
        logger.info(f"Models saved to: {self.save_dir}")

        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                progress_bar=True,
            )

            # Save final model
            final_path = os.path.join(self.save_dir, "model_final")
            self.model.save(final_path)
            logger.info(f"Training complete. Final model: {final_path}")

            # Save final metrics
            metrics_logger.save()

        except KeyboardInterrupt:
            self._save_interrupt()
        finally:
            signal.signal(signal.SIGINT, original_sigint)

    def _save_interrupt(self):
        """Save model on interrupt."""
        if self.model:
            path = os.path.join(self.save_dir, "model_interrupt")
            self.model.save(path)
            logger.info(f"Interrupt model saved: {path}")
