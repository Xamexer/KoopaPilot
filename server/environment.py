"""Gymnasium-compatible environment wrapping BizHawk socket communication."""

import logging
import threading
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .games.smw.actions import DISCRETE_ACTIONS
from .observation import build_observation, get_observation_size
from .reward import EpisodeTracker, compute_reward, init_tracker_from_state

logger = logging.getLogger(__name__)

NO_OP_RESPONSE = {
    "type": "action",
    "action": [0, 0, 0, 0, 0, 0, 1],
    "total_reward": 0.0,
    "reward_event": "",
}


class SMWEnvironment(gym.Env):
    """Single SMW environment backed by a BizHawk emulator via sockets.

    Communication flow:
    - Before reset(): on_state_received() responds immediately with no-op
      (emulators send states right after connecting, before training starts)
    - After reset(): on_state_received() blocks until step()/reset() provides action

    Thread safety: _lock protects all shared state transitions.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, emulator_id: int, config: dict):
        super().__init__()
        self.emulator_id = emulator_id
        self.config = config

        obs_size = get_observation_size(config)
        self.observation_space = spaces.Box(
            low=-0.5, high=1.5, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(DISCRETE_ACTIONS))

        self.tracker = EpisodeTracker()
        self._last_obs = np.zeros(obs_size, dtype=np.float32)
        self._done = False

        # Synchronization between socket thread and RL thread
        self._lock = threading.Lock()
        self._state_ready = threading.Event()
        self._action_ready = threading.Event()
        self._pending_state = None
        self._pending_response = None
        self.last_state = None
        self.last_transition_state = None

        # Before first reset(), respond immediately so emulator startup cannot
        # block while the remaining instances and the agent are initialized.
        self._initialized = False

    def on_state_received(self, state: dict) -> dict:
        """Called by socket server when a state message arrives.

        Before first reset(): responds immediately with all buttons released.
        After reset(): blocks until step()/reset() provides an action.
        """
        # Before training starts, don't block or send an input-bearing action.
        if not self._initialized:
            return NO_OP_RESPONSE.copy()

        # Normal flow: store state, signal ready, wait for action
        with self._lock:
            self._pending_state = state
            self._state_ready.set()

        if not self._action_ready.wait(timeout=30.0):
            logger.warning(f"[Env {self.emulator_id}] Action timeout")
            return NO_OP_RESPONSE.copy()

        with self._lock:
            self._action_ready.clear()
            response = self._pending_response or NO_OP_RESPONSE.copy()
            return response

    def step(self, action):
        """Standard gym step. Sends action, waits for next state."""
        # Convert discrete action index to button list
        action_idx = int(action)
        action_list = DISCRETE_ACTIONS[action_idx]

        # Respond to the PREVIOUS state with this action
        # Use the last event from previous step (events are calculated after state is received)
        with self._lock:
            self._pending_response = {
                "type": "action",
                "action": action_list,
                "total_reward": self.tracker.total_reward,
                "reward_event": self.tracker.last_event,
            }
            self._action_ready.set()

        # Wait for the NEXT state from emulator
        if not self._state_ready.wait(timeout=30.0):
            logger.warning(f"[Env {self.emulator_id}] State timeout in step()")
            return self._last_obs, 0.0, True, False, {}

        with self._lock:
            self._state_ready.clear()
            state = self._pending_state

        if state is None:
            return self._last_obs, 0.0, True, False, {}

        self.last_state = state
        self.last_transition_state = state

        # Compute observation and reward
        obs = build_observation(state, self.config)
        reward, event_str, done = compute_reward(state, self.tracker, self.config)
        truncated = done and self.tracker.truncated
        terminated = done and not truncated

        self._last_obs = obs
        self._done = done

        info = {
            "total_reward": self.tracker.total_reward,
            "mario_x": state.get("mario_x_level", 0),
            "goal_reached": self.tracker.goal_reached,
        }
        if truncated:
            info["TimeLimit.truncated"] = True
        if event_str:
            info["reward_event"] = event_str
        
        # Add episode info when done (for SB3 callback)
        if done:
            info["episode"] = {
                "r": self.tracker.total_reward,
                "l": state.get("step", 0),
                "max_x": self.tracker.max_x,
                "goal_reached": self.tracker.goal_reached,
                "truncated": truncated,
            }

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        """Reset the episode. Sends reset command to emulator."""
        super().reset(seed=seed)
        self.tracker = EpisodeTracker()
        self._done = False

        if not self._initialized:
            # First reset: take control of the next emulator state, explicitly
            # send a reset, then wait for the freshly loaded state. Previously
            # the reset response was prepared but never released to Lua.
            with self._lock:
                self._state_ready.clear()
                self._initialized = True
            logger.info(f"[Env {self.emulator_id}] Initialized")

            if not self._state_ready.wait(timeout=30.0):
                logger.warning(
                    f"[Env {self.emulator_id}] State timeout before first reset()"
                )
                return self._last_obs, {}

            # Release the waiting state with a reset command. Lua performs the
            # configured savestate or full-level load before sending again.
            with self._lock:
                self._state_ready.clear()
                self._pending_response = {"type": "reset"}
                self._action_ready.set()

            if not self._state_ready.wait(timeout=30.0):
                logger.warning(
                    f"[Env {self.emulator_id}] State timeout after first reset()"
                )
                with self._lock:
                    self._pending_response = NO_OP_RESPONSE.copy()
                    self._action_ready.set()
                return self._last_obs, {}

            with self._lock:
                self._state_ready.clear()
                state = self._pending_state
        else:
            # Subsequent resets: release the current waiting state with a reset
            # command, then wait for the freshly loaded state.
            with self._lock:
                self._pending_response = {"type": "reset"}
                self._action_ready.set()
                self._state_ready.clear()

            if not self._state_ready.wait(timeout=30.0):
                logger.warning(f"[Env {self.emulator_id}] State timeout in reset()")
                with self._lock:
                    self._pending_response = NO_OP_RESPONSE.copy()
                    self._action_ready.set()
                return self._last_obs, {}

            with self._lock:
                self._state_ready.clear()
                state = self._pending_state

        if state:
            self.last_state = state
            obs = build_observation(state, self.config)
            # Initialize tracker with state values to prevent false rewards
            init_tracker_from_state(self.tracker, state, self.config)
            self._last_obs = obs

        # Leave Lua blocked on the freshly loaded state until step() supplies
        # the policy's first real action. Sending a no-op here inserted one
        # extra frame-skip interval that RetroJet never executes.

        return self._last_obs, {}
