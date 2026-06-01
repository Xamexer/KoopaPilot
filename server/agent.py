"""PPO agent wrapper using stable-baselines3."""

import os
import logging
from typing import Optional
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import FloatSchedule

logger = logging.getLogger(__name__)


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
                 metrics_logger=None, verbose=1):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.save_interval = save_interval  # In timesteps, not calls
        self.metrics_logger = metrics_logger
        self.best_mean_reward = float("-inf")
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_max_x = []
        self.last_log_timestep = 0
        self.last_save_timestep = 0

    def _on_step(self) -> bool:
        # Track episode info
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                self.episode_max_x.append(info["episode"].get("max_x", 0))

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
                
                self.metrics_logger.log_iteration({
                    "timestep": self.num_timesteps,
                    "mean_reward": float(mean_rew),
                    "max_reward": float(max_rew),
                    "min_reward": float(min_rew),
                    "mean_length": float(mean_len),
                    "mean_max_x": float(mean_max_x),
                    "max_x": float(max_x),
                    "episodes": len(self.episode_rewards),
                })
                


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
