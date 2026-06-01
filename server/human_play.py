"""Human play mode: let the player control Mario while showing rewards."""

import logging
from .reward import EpisodeTracker, compute_reward

logger = logging.getLogger(__name__)


class HumanPlayManager:
    """Handles human play mode where the player controls the game
    and reward values are displayed on the emulator overlay."""

    def __init__(self, config: dict):
        self.config = config
        self.trackers: dict[int, EpisodeTracker] = {}
        self.running = True

    def on_state(self, emulator_id: int, state: dict) -> dict:
        """Process state from emulator in human play mode.

        Does NOT send actions - the player controls via real controller.
        Only computes and displays rewards.
        """
        if emulator_id not in self.trackers:
            self.trackers[emulator_id] = EpisodeTracker()

        tracker = self.trackers[emulator_id]

        # Check for death/goal to reset tracker
        player_anim = state.get("player_anim", 0)
        if player_anim == 0x09:  # Death
            reward, event, _ = compute_reward(state, tracker, self.config)
            response = {
                "type": "action",
                "action": [0, 0, 0, 0, 0, 0, 0],  # No override
                "total_reward": tracker.total_reward,
                "reward_event": event,
            }
            # Reset tracker for next life
            self.trackers[emulator_id] = EpisodeTracker()
            return response

        if player_anim in (0x0A, 0x0B):  # Goal
            reward, event, _ = compute_reward(state, tracker, self.config)
            response = {
                "type": "action",
                "action": [0, 0, 0, 0, 0, 0, 0],
                "total_reward": tracker.total_reward,
                "reward_event": event,
            }
            self.trackers[emulator_id] = EpisodeTracker()
            return response

        # Normal state
        reward, event, done = compute_reward(state, tracker, self.config)

        return {
            "type": "action",
            "action": [0, 0, 0, 0, 0, 0, 0],  # Don't override player input
            "total_reward": tracker.total_reward,
            "reward_event": event,
        }

    def reset_tracker(self, emulator_id: int):
        """Reset reward tracking for an emulator."""
        self.trackers[emulator_id] = EpisodeTracker()
