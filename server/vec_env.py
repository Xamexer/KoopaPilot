"""Vectorized environment for parallel BizHawk emulators."""

import logging
import numpy as np
from typing import Optional
from stable_baselines3.common.vec_env import VecEnv

from .observation import get_observation_size
from .environment import SMWEnvironment

logger = logging.getLogger(__name__)


class SMWVecEnv(VecEnv):
    """Vectorized environment using socket-connected BizHawk emulators.

    Unlike SubprocVecEnv, this doesn't use subprocesses - the emulators
    are external BizHawk processes communicating via TCP sockets.
    Each environment runs its step synchronously with its emulator.
    """

    def __init__(self, envs: list[SMWEnvironment]):
        self.envs = envs
        num_envs = len(envs)
        obs_size = get_observation_size(envs[0].config)

        observation_space = envs[0].observation_space
        action_space = envs[0].action_space

        super().__init__(num_envs, observation_space, action_space)

        self._observations = np.zeros(
            (num_envs, obs_size), dtype=np.float32
        )
        self._rewards = np.zeros(num_envs, dtype=np.float32)
        self._dones = np.zeros(num_envs, dtype=bool)
        self._infos = [{} for _ in range(num_envs)]
        self._actions: Optional[np.ndarray] = None

    def reset(self):
        """Reset all environments."""
        import threading
        threads = []
        results = [None] * self.num_envs

        def _reset_env(idx):
            obs, info = self.envs[idx].reset()
            results[idx] = obs

        for i in range(self.num_envs):
            t = threading.Thread(target=_reset_env, args=(i,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join(timeout=60.0)

        for i in range(self.num_envs):
            if results[i] is not None:
                self._observations[i] = results[i]

        return self._observations.copy()

    def step_async(self, actions):
        """Store actions to be executed."""
        self._actions = actions

    def step_wait(self):
        """Execute stored actions and wait for results."""
        import threading

        if self._actions is None:
            raise RuntimeError("step_async must be called before step_wait")

        threads = []
        results = [None] * self.num_envs

        def _step_env(idx, action):
            try:
                obs, reward, terminated, truncated, info = self.envs[idx].step(action)
                done = terminated or truncated
                info["TimeLimit.truncated"] = truncated and not terminated
                if done:
                    reset_obs, _ = self.envs[idx].reset()
                    info["terminal_observation"] = obs
                    obs = reset_obs
                results[idx] = (obs, reward, done, info)
            except Exception as e:
                logger.error(f"Error in env {idx}: {e}")
                obs_size = get_observation_size(self.envs[idx].config)
                results[idx] = (
                    np.zeros(obs_size, dtype=np.float32),
                    0.0, True, {"error": str(e)}
                )

        for i in range(self.num_envs):
            t = threading.Thread(
                target=_step_env, args=(i, self._actions[i])
            )
            t.start()
            threads.append(t)

        for t in threads:
            t.join(timeout=60.0)

        for i in range(self.num_envs):
            if results[i] is not None:
                obs, rew, done, info = results[i]
                self._observations[i] = obs
                self._rewards[i] = rew
                self._dones[i] = done
                self._infos[i] = info

        self._actions = None
        return (
            self._observations.copy(),
            self._rewards.copy(),
            self._dones.copy(),
            self._infos.copy()
        )

    def close(self):
        """Close all environments."""
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        target_envs = self._get_target_envs(indices)
        return [
            getattr(env, method_name)(*method_args, **method_kwargs)
            for env in target_envs
        ]

    def get_attr(self, attr_name, indices=None):
        target_envs = self._get_target_envs(indices)
        return [getattr(env, attr_name) for env in target_envs]

    def set_attr(self, attr_name, value, indices=None):
        target_envs = self._get_target_envs(indices)
        for env in target_envs:
            setattr(env, attr_name, value)

    def seed(self, seed=None):
        return [None] * self.num_envs

    def _get_target_envs(self, indices):
        if indices is None:
            return self.envs
        return [self.envs[i] for i in indices]

    def _get_indices(self, indices):
        if indices is None:
            return list(range(self.num_envs))
        return indices
