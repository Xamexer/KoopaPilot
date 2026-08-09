"""PPO agent wrapper using stable-baselines3."""

import json
import logging
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Optional
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import FloatSchedule

logger = logging.getLogger(__name__)
LIVE_MODEL_IO_LOCK = threading.RLock()


def linear_schedule(initial_value: float):
    """Linear learning rate schedule."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def create_agent(vec_env, config: dict, model_path: Optional[str] = None) -> PPO:
    """Create or load a PPO agent."""
    ppo_cfg = config.get("ppo", {})
    policy_kwargs = ppo_cfg.get("policy_kwargs", {"net_arch": [256, 256]})

    # Convert net_arch dict format for SB3 (pi/vf separate networks)
    net_arch = policy_kwargs.get("net_arch", [256, 256])
    if isinstance(net_arch, dict):
        policy_kwargs["net_arch"] = dict(
            pi=net_arch.get("pi", [256, 256]),
            vf=net_arch.get("vf", [256, 256]),
        )

    lr = ppo_cfg.get("learning_rate", 2.5e-4)
    lr_schedule = ppo_cfg.get("lr_schedule", "linear")
    if lr_schedule == "linear":
        lr = linear_schedule(lr)

    clip_range = ppo_cfg.get("clip_range", 0.2)
    clip_range_schedule = ppo_cfg.get("clip_range_schedule", "constant")
    if clip_range_schedule == "linear":
        clip_range = linear_schedule(clip_range)

    if model_path and os.path.exists(model_path):
        logger.info(f"Loading model from {model_path}")
        try:
            model = PPO.load(model_path, env=vec_env)
        except ValueError as exc:
            raise ValueError(
                "Checkpoint is incompatible with the current observation or "
                "action space. Start a fresh run after changing observations "
                "or controller actions."
            ) from exc
        # Checkpoints retain the TensorBoard directory from their original
        # machine. Always redirect new logs to the active configuration.
        model.tensorboard_log = config.get("paths", {}).get(
            "log_dir", "./logs"
        )
        # Apply current config hyperparameters (important for finetuning)
        model.learning_rate = lr
        model.lr_schedule = FloatSchedule(lr)
        model.ent_coef = ppo_cfg.get("ent_coef", 0.01)
        model.clip_range = FloatSchedule(clip_range)
        configured_n_steps = ppo_cfg.get("n_steps", 2048)
        if configured_n_steps != model.n_steps:
            raise ValueError(
                f"Checkpoint uses n_steps={model.n_steps}, but config requests "
                f"n_steps={configured_n_steps}. Start a fresh run or use a "
                "matching config so PPO does not silently keep an old rollout "
                "buffer size."
            )
        model.batch_size = ppo_cfg.get("batch_size", 64)
        model.n_epochs = ppo_cfg.get("n_epochs", 4)
        model.gamma = ppo_cfg.get("gamma", 0.99)
        model.gae_lambda = ppo_cfg.get("gae_lambda", 0.95)
        model.vf_coef = ppo_cfg.get("vf_coef", 0.5)
        model.max_grad_norm = ppo_cfg.get("max_grad_norm", 0.5)
        model.target_kl = ppo_cfg.get("target_kl")
        model.rollout_buffer.gamma = model.gamma
        model.rollout_buffer.gae_lambda = model.gae_lambda
        logger.info(
            "Applied config: lr=%s, ent_coef=%s, gamma=%s, "
            "gae_lambda=%s, lr_schedule=%s, clip_schedule=%s",
            ppo_cfg.get("learning_rate"), model.ent_coef, model.gamma,
            model.gae_lambda, lr_schedule, clip_range_schedule,
        )
        return model

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=lr,
        n_steps=ppo_cfg.get("n_steps", 2048),
        batch_size=ppo_cfg.get("batch_size", 64),
        n_epochs=ppo_cfg.get("n_epochs", 4),
        gamma=ppo_cfg.get("gamma", 0.99),
        gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
        clip_range=clip_range,
        ent_coef=ppo_cfg.get("ent_coef", 0.01),
        vf_coef=ppo_cfg.get("vf_coef", 0.5),
        max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
        target_kl=ppo_cfg.get("target_kl"),
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=config["paths"].get("log_dir", "./logs"),
    )

    logger.info("Created new PPO agent")
    return model


class CheckpointCallback(BaseCallback):
    """Save model at regular intervals and track best reward."""

    def __init__(self, save_dir: str, save_interval: int,
                 metrics_logger=None, live_model_path: Optional[str] = None,
                 live_save_interval: int = 10_000, verbose=1):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.save_interval = save_interval  # In timesteps, not calls
        self.metrics_logger = metrics_logger
        self.live_model_path = live_model_path
        self.live_save_interval = max(1, int(live_save_interval))
        self.best_mean_reward = float("-inf")
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_max_x = []
        self.episode_goals = []
        self.last_log_timestep = 0
        self.last_save_timestep = 0
        self.last_live_save_timestep = 0

    def _on_step(self) -> bool:
        # Track episode info
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                self.episode_max_x.append(info["episode"].get("max_x", 0))
                self.episode_goals.append(bool(
                    info["episode"].get("goal_reached", False)
                ))

        # Log metrics every 2048 global steps for live dashboard updates
        log_interval = 2048
        if self.num_timesteps - self.last_log_timestep >= log_interval:
            self.last_log_timestep = self.num_timesteps
            if len(self.episode_rewards) > 0 and self.metrics_logger:
                import numpy as np
                mean_rew = np.mean(self.episode_rewards[-50:])
                max_rew = np.max(self.episode_rewards[-50:])
                min_rew = np.min(self.episode_rewards[-50:])
                mean_len = np.mean(self.episode_lengths[-50:])
                mean_max_x = np.mean(self.episode_max_x[-50:])
                max_x = np.max(self.episode_max_x[-50:])
                goal_rate = np.mean(self.episode_goals[-50:])
                
                self.metrics_logger.log_iteration({
                    "timestep": self.num_timesteps,
                    "mean_reward": float(mean_rew),
                    "max_reward": float(max_rew),
                    "min_reward": float(min_rew),
                    "mean_length": float(mean_len),
                    "mean_max_x": float(mean_max_x),
                    "max_x": float(max_x),
                    "goal_rate": float(goal_rate),
                    "episodes": len(self.episode_rewards),
                })
        if (
            self.live_model_path
            and self.num_timesteps - self.last_live_save_timestep
            >= self.live_save_interval
        ):
            self.last_live_save_timestep = self.num_timesteps
            try:
                published_path = _save_model_atomic(
                    self.model,
                    self.live_model_path,
                    timestep=self.num_timesteps,
                )
                if self.verbose:
                    logger.info(
                        f"Published live demo model: {published_path}"
                    )
            except Exception as exc:
                # The visible viewer is optional. A temporary Windows file
                # lock, antivirus scan, or disk error must not kill training.
                logger.warning(
                    "Could not refresh live demo model; training continues "
                    "with the previous viewer checkpoint: %s",
                    exc,
                )

        # Save model at intervals (based on timesteps, not calls)
        if self.num_timesteps - self.last_save_timestep >= self.save_interval:
            self.last_save_timestep = self.num_timesteps
            
            # Save checkpoint
            path = os.path.join(self.save_dir, f"model_step_{self.num_timesteps}")
            self.model.save(path)
            if self.verbose:
                logger.info(f"Saved checkpoint: {path}")

            # Track and save best model
            if len(self.episode_rewards) > 0:
                import numpy as np
                mean_rew = np.mean(self.episode_rewards[-100:])
                if mean_rew > self.best_mean_reward:
                    self.best_mean_reward = mean_rew
                    best_path = os.path.join(self.save_dir, "model_best")
                    self.model.save(best_path)
                    logger.info(
                        f"New best model! Mean reward: {mean_rew:.2f}"
                    )

        return True


def _save_model_atomic(model, path: str, timestep: int | None = None):
    """Publish a model without replacing a ZIP that Windows may be reading."""
    with LIVE_MODEL_IO_LOCK:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        target_path = (
            _generation_model_path(path, timestep)
            if timestep is not None
            else path
        )
        fd, temp_path = tempfile.mkstemp(
            prefix=".model-live-",
            suffix=".zip",
            dir=os.path.dirname(path) or ".",
        )
        os.close(fd)
        try:
            model.save(temp_path)
            _replace_with_retry(temp_path, target_path)
            if timestep is not None:
                _save_model_metadata_atomic(path, target_path, timestep)
                _cleanup_model_generations(path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        return target_path


def _model_metadata_path(model_path: str) -> str:
    path = os.path.abspath(model_path)
    if path.lower().endswith(".zip"):
        path = path[:-4]
    return f"{path}.meta.json"


def _save_model_metadata_atomic(
    reference_path: str, model_path: str, timestep: int
):
    """Write metadata that can be verified against the completed model file."""
    model_stat = os.stat(model_path)
    metadata_path = _model_metadata_path(reference_path)
    fd, temp_path = tempfile.mkstemp(
        prefix=".model-live-meta-",
        suffix=".json",
        dir=os.path.dirname(metadata_path) or ".",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({
                "timestep": int(timestep),
                "model_file": os.path.basename(model_path),
                "model_mtime_ns": model_stat.st_mtime_ns,
                "model_size": model_stat.st_size,
            }, handle, indent=2)
        _replace_with_retry(temp_path, metadata_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _generation_model_path(reference_path: str, timestep: int) -> str:
    """Return a new immutable filename for one live-model generation."""
    path = Path(os.path.abspath(reference_path))
    stem = path.name[:-4] if path.name.lower().endswith(".zip") else path.name
    name = (
        f"{stem}.generation-{int(timestep):012d}-"
        f"{uuid.uuid4().hex[:8]}.zip"
    )
    return str(path.with_name(name))


def _cleanup_model_generations(
    reference_path: str, keep_latest: int = 4
):
    """Best-effort cleanup of superseded immutable live-model files."""
    path = Path(os.path.abspath(reference_path))
    stem = path.name[:-4] if path.name.lower().endswith(".zip") else path.name
    candidates = sorted(
        path.parent.glob(f"{stem}.generation-*.zip"),
        key=lambda candidate: candidate.stat().st_mtime_ns,
        reverse=True,
    )
    for candidate in candidates[max(0, keep_latest):]:
        try:
            candidate.unlink()
        except OSError:
            # A stale generation is harmless and will be retried next time.
            pass


def _replace_with_retry(
    source: str,
    destination: str,
    attempts: int = 80,
    delay_seconds: float = 0.025,
):
    """Replace a file, tolerating short-lived Windows reader locks."""
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay_seconds)
