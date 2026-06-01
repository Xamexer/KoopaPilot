"""Reward calculation from game state deltas."""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EpisodeTracker:
    """Tracks per-episode state for reward computation."""
    max_x: float = 0.0
    max_y: float = 0.0  # For vertical levels: lowest Y value = highest point
    last_x: float = 0.0
    last_y: float = 0.0
    last_coins: int = 0
    last_powerup: int = 0
    last_lives: int = 0
    last_sublevel: int = 0
    last_sprite_status: list = field(default_factory=lambda: [0] * 12)
    total_reward: float = 0.0
    steps_without_progress: int = 0
    is_first_step: bool = True
    last_event: str = ""  # Store last reward event for display
    goal_reached: bool = False
    truncated: bool = False


def init_tracker_from_state(tracker: EpisodeTracker, state: dict):
    """Initialize tracker with current state values (for reset/teleport).
    This prevents false rewards when loading a savestate mid-level."""
    tracker.last_x = state.get("mario_x_level", 0)
    tracker.last_y = state.get("mario_y_level", 0)
    tracker.max_x = tracker.last_x
    tracker.max_y = tracker.last_y
    tracker.last_coins = state.get("coins", 0)
    tracker.last_powerup = state.get("powerup", 0)
    tracker.last_lives = state.get("lives", 0)
    tracker.last_sublevel = state.get("sublevel", 0)
    tracker.last_event = "TELEPORT/RESET"
    tracker.is_first_step = False  # Mark as initialized
    tracker.steps_without_progress = 0
    
    # Read initial sprite status
    sprites = state.get("sprites", [])
    for i in range(min(12, len(sprites))):
        tracker.last_sprite_status[i] = sprites[i].get("status", 0)


def compute_reward(state: dict, tracker: EpisodeTracker,
                   config: dict) -> tuple[float, str, bool]:
    """Compute reward from state and tracker.

    Returns:
        (reward, event_description, is_done)
    """
    rewards_cfg = config.get("rewards", {})
    ppo_cfg = config.get("ppo", {})

    reward = 0.0
    events = []
    done = False
    tracker.truncated = False

    mario_x = state.get("mario_x_level", 0)
    mario_y = state.get("mario_y_level", 0)
    coins = state.get("coins", 0)
    powerup = state.get("powerup", 0)
    lives = state.get("lives", 0)
    player_anim = state.get("player_anim", 0)
    sublevel = state.get("sublevel", 0)
    is_vertical = state.get("is_vertical", False)
    
    # Initialize on first step
    if tracker.is_first_step:
        tracker.last_x = mario_x
        tracker.last_y = mario_y
        tracker.max_x = mario_x
        tracker.max_y = mario_y
        tracker.last_coins = coins
        tracker.last_powerup = powerup
        tracker.last_lives = lives
        tracker.last_sublevel = sublevel
        tracker.is_first_step = False

        # Read initial sprite status
        sprites = state.get("sprites", [])
        for i in range(min(12, len(sprites))):
            tracker.last_sprite_status[i] = sprites[i].get("status", 0)
        return 0.0, "", False

    # --- DEATH CHECK ---
    if player_anim == 0x09:
        penalty = rewards_cfg.get("death_penalty", -500.0)
        reward += penalty
        events.append(f"DEATH {penalty:.0f}")
        done = True
        tracker.total_reward += reward
        return reward, " | ".join(events), done

    # --- GOAL CHECK ---
    # Check both player_anim (0x0A, 0x0B) and goal_reached flag (0x1493)
    goal_flag = state.get("goal_reached", 0)
    if player_anim in (0x0A, 0x0B) or goal_flag != 0:
        bonus = rewards_cfg.get("goal_reached", 1000.0)
        reward += bonus
        events.append(f"GOAL! +{bonus:.0f}")
        done = True
        tracker.goal_reached = True
        tracker.total_reward += reward
        logger.info(
            "Goal detected: player_anim=0x%02X, goal_flag=%s, reward=%s",
            player_anim, goal_flag, reward,
        )
        return reward, " | ".join(events), done

    # --- TELEPORT DETECTION (pipe/door/savestate load) ---
    threshold = rewards_cfg.get("teleport_threshold_pixels", 256)
    x_diff = abs(mario_x - tracker.last_x)
    teleported = x_diff > threshold

    # Sublevel change
    if sublevel != tracker.last_sublevel:
        pipe_reward = rewards_cfg.get("pipe_door_entered", 100.0)
        reward += pipe_reward
        events.append(f"PIPE/DOOR +{pipe_reward:.0f}")
        # Reset tracking for new area
        tracker.max_x = mario_x
        tracker.max_y = mario_y
        tracker.last_sublevel = sublevel
        tracker.steps_without_progress = 0
    elif teleported:
        # Large X jump without sublevel change (e.g., savestate load mid-level)
        # Treat this as a new starting position - no reward for the jump
        tracker.last_x = mario_x
        tracker.last_y = mario_y
        tracker.max_x = mario_x
        tracker.max_y = mario_y
        tracker.steps_without_progress = 0
        events.append(f"TELEPORT {x_diff:.0f}px")

    # --- X PROGRESS (only for new max X) ---
    # Rewarding raw delta-X can be exploited by moving back and forth because
    # PPO discounts the later negative leg of the cycle. A monotonic record
    # keeps the dense progress signal without making oscillation profitable.
    made_progress = False
    if not teleported:
        if mario_x > tracker.max_x:
            max_delta = mario_x - tracker.max_x
            reward_rate = rewards_cfg.get("x_progress_per_pixel", 0.25)
            x_reward = max_delta * reward_rate
            reward += x_reward
            tracker.max_x = mario_x
            made_progress = True
            events.append(f"NEW_X +{x_reward:.1f}")

    # --- Y PROGRESS (vertical levels) ---
    if is_vertical and not teleported:
        # Y=0 is top, so going up means Y decreases.
        if mario_y < tracker.max_y:
            y_delta = tracker.max_y - mario_y
            y_reward = y_delta * rewards_cfg.get("y_progress_per_pixel", 0.05)
            reward += y_reward
            tracker.max_y = mario_y
            made_progress = True
            events.append(f"NEW_Y +{y_reward:.1f}")

    # --- STAGNATION TRACKING ---
    if not teleported:
        if made_progress:
            tracker.steps_without_progress = 0
        else:
            tracker.steps_without_progress += 1

    # --- COINS ---
    # Skip coin rewards right after teleport/reset to prevent false rewards from savestate
    coin_delta = coins - tracker.last_coins
    if coin_delta < 0:
        coin_delta += 100  # Wrap from 99->0
    if coin_delta > 0 and tracker.last_event != "TELEPORT/RESET":
        coin_reward = coin_delta * rewards_cfg.get("coin_collected", 5.0)
        reward += coin_reward
        events.append(f"COIN +{coin_reward:.0f}")

    # --- POWERUP ---
    # Skip powerup rewards right after teleport/reset to prevent false rewards from savestate
    if powerup > tracker.last_powerup and tracker.last_event != "TELEPORT/RESET":
        pu_reward = rewards_cfg.get("powerup_upgrade", 50.0)
        reward += pu_reward
        events.append(f"POWERUP +{pu_reward:.0f}")
    elif powerup < tracker.last_powerup and player_anim != 0x09:
        pu_penalty = rewards_cfg.get("powerup_downgrade", -100.0)
        reward += pu_penalty
        events.append(f"HIT {pu_penalty:.0f}")

    # --- LIVES (1-UP) ---
    # Skip 1UP rewards right after teleport/reset to prevent false rewards from savestate
    life_delta = lives - tracker.last_lives
    if life_delta > 0 and player_anim != 0x09 and tracker.last_event != "TELEPORT/RESET":
        oneup_reward = life_delta * rewards_cfg.get("oneup_collected", 25.0)
        reward += oneup_reward
        events.append(f"1UP +{oneup_reward:.0f}")

    # --- ENEMY KILLS ---
    # Skip enemy kill rewards right after teleport/reset to prevent false positives
    skip_enemy_rewards = tracker.last_event == "TELEPORT/RESET"
    sprites = state.get("sprites", [])
    kills = 0
    stuns = 0
    for i in range(min(12, len(sprites))):
        prev = tracker.last_sprite_status[i]
        curr = sprites[i].get("status", 0)
        # Sprite is considered alive if status >= 0x08 (active)
        was_alive = prev >= 0x08
        # Sprite is considered killed if it goes to specific death/stunned states
        is_killed = curr in (0x02, 0x03, 0x04, 0x05, 0x06, 0x0A)
        is_stunned = curr == 0x09
        # Sprite despawned/offscreen
        is_despawned = curr == 0x00 or curr < 0x08

        # Reset tracker if sprite despawned or ID changed significantly
        # This prevents false positives when new sprites spawn in old slots
        if was_alive and is_killed:
            kills += 1
            tracker.last_sprite_status[i] = curr
        elif was_alive and is_stunned and prev != 0x09:
            stuns += 1
            tracker.last_sprite_status[i] = curr
        elif is_despawned and prev > 0:
            # Sprite went offscreen/despawned - reset tracker for this slot
            # so we don't count it as a kill when a new sprite spawns here
            tracker.last_sprite_status[i] = 0
        else:
            tracker.last_sprite_status[i] = curr

    # Cap kills/stuns per step to prevent reward spam
    kills = min(kills, 3)  # Max 3 kills per step
    stuns = min(stuns, 3)  # Max 3 stuns per step
    

    
    # Skip enemy rewards right after teleport/reset
    if kills > 0 and not skip_enemy_rewards:
        kill_reward = kills * rewards_cfg.get("enemy_killed", 25.0)
        reward += kill_reward
        events.append(f"KILL x{kills} +{kill_reward:.0f}")
    if stuns > 0 and not skip_enemy_rewards:
        stun_reward = stuns * rewards_cfg.get("enemy_stunned", 10.0)
        reward += stun_reward
        events.append(f"STUN x{stuns} +{stun_reward:.0f}")

    # --- TIME PENALTY ---
    time_penalty = rewards_cfg.get("time_penalty_per_step", -0.1)
    reward += time_penalty

    # --- STAGNATION TIMEOUT ---
    stagnation_limit = ppo_cfg.get("stagnation_timeout_steps", 600)
    if tracker.steps_without_progress >= stagnation_limit:
        events.append("TIMEOUT")
        done = True
        tracker.truncated = True

    # --- MAX EPISODE STEPS ---
    step = state.get("step", 0)
    max_steps = ppo_cfg.get("max_episode_steps", 4500)
    if step >= max_steps:
        events.append("MAX_STEPS")
        done = True
        tracker.truncated = True

    # Update tracker
    tracker.last_x = mario_x
    tracker.last_y = mario_y
    tracker.last_coins = coins
    tracker.last_powerup = powerup
    tracker.last_lives = lives
    tracker.total_reward += reward

    event_str = " | ".join(events) if events else ""
    tracker.last_event = event_str  # Store for display in Lua
    return reward, event_str, done
