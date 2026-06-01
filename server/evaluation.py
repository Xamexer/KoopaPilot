"""Evaluation mode: run trained model with live viewing and video recording."""

import logging
import os
import glob
import shutil
import time
import numpy as np
from stable_baselines3 import PPO

logger = logging.getLogger(__name__)


def compile_video(frames_dir: str, output_path: str, fps: int = 15):
    """Compile PNG frames into an MP4 video using OpenCV."""
    import cv2

    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if not frame_files:
        logger.warning(f"No frames found in {frames_dir}")
        return False

    # Read first frame to get dimensions
    first = cv2.imread(frame_files[0])
    if first is None:
        logger.warning(f"Could not read frame: {frame_files[0]}")
        return False

    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for f in frame_files:
        img = cv2.imread(f)
        if img is not None:
            writer.write(img)

    writer.release()
    logger.info(f"Video saved: {output_path} ({len(frame_files)} frames, {len(frame_files)/fps:.1f}s)")
    return True


class EvaluationManager:
    """Run evaluation episodes with live emulator viewing and video recording."""

    def __init__(self, vec_env, config: dict, model_path: str):
        self.vec_env = vec_env
        self.config = config
        self.model_path = model_path

    def evaluate(self, num_episodes: int = 3):
        """Run evaluation episodes."""
        if not os.path.exists(self.model_path):
            logger.error(f"Model not found: {self.model_path}")
            return

        model = PPO.load(self.model_path, env=self.vec_env)
        video_dir = self.config["paths"].get("video_dir", "./videos")
        os.makedirs(video_dir, exist_ok=True)

        results = []

        logger.info(f"Starting evaluation: {num_episodes} episodes")
        logger.info(f"Model: {self.model_path}")
        logger.info("=" * 50)

        for ep in range(num_episodes):
            logger.info(f"\n--- Episode {ep + 1}/{num_episodes} ---")

            obs = self.vec_env.reset()
            total_reward = 0.0
            steps = 0
            done = False
            max_x = 0
            goal_reached = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, dones, infos = self.vec_env.step(action)
                total_reward += reward[0]
                steps += 1
                done = dones[0]

                # Track max x from info
                if infos and len(infos) > 0:
                    info = infos[0]
                    x = info.get("mario_x", 0)
                    if x > max_x:
                        max_x = x
                    goal_reached = goal_reached or info.get("goal_reached", False)

                # Progress logging every 100 steps
                if steps % 100 == 0:
                    logger.info(
                        f"  Step {steps}: reward={total_reward:.1f}, max_x={max_x}"
                    )

            result = {
                "episode": ep + 1,
                "total_reward": float(total_reward),
                "steps": steps,
                "max_x": max_x,
                "goal_reached": goal_reached,
            }
            results.append(result)

            status = "GOAL!" if goal_reached else "DIED/TIMEOUT"
            logger.info(
                f"  => {status} | reward={total_reward:.1f}, "
                f"steps={steps}, max_x={max_x}"
            )

            # Compile video from screenshots if available
            frames_dir = self.config.get("_screenshot_dir")
            if frames_dir and os.path.isdir(frames_dir):
                video_path = os.path.join(video_dir, f"eval_ep{ep + 1}.mp4")
                try:
                    if compile_video(frames_dir, video_path):
                        result["video"] = video_path
                except Exception as e:
                    logger.warning(f"Video compilation failed: {e}")
                # Clean up frames
                try:
                    shutil.rmtree(frames_dir)
                    os.makedirs(frames_dir, exist_ok=True)
                except Exception:
                    pass

            # Pause between episodes
            if ep < num_episodes - 1:
                logger.info("  Next episode in 3 seconds...")
                time.sleep(3)

        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 50)
        mean_reward = np.mean([r["total_reward"] for r in results])
        goals = sum(1 for r in results if r["goal_reached"])
        logger.info(f"  Episodes: {num_episodes}")
        logger.info(f"  Goals reached: {goals}/{num_episodes}")
        logger.info(f"  Mean reward: {mean_reward:.1f}")
        logger.info(f"  Best reward: {max(r['total_reward'] for r in results):.1f}")
        logger.info(f"  Mean steps: {np.mean([r['steps'] for r in results]):.0f}")
        for r in results:
            status = "GOAL" if r["goal_reached"] else "FAIL"
            video_info = f", video={r['video']}" if "video" in r else ""
            logger.info(
                f"  Ep {r['episode']}: [{status}] "
                f"reward={r['total_reward']:.1f}, "
                f"steps={r['steps']}, max_x={r['max_x']}{video_info}"
            )

        # Save results to file
        self._save_results(video_dir, results, mean_reward)
        return results

    def _save_results(self, video_dir: str, results: list, mean_reward: float):
        """Save evaluation results to a text file."""
        path = os.path.join(video_dir, "eval_results.txt")
        with open(path, "w") as f:
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Episodes: {len(results)}\n")
            f.write(f"Mean reward: {mean_reward:.2f}\n")
            f.write(f"Goals: {sum(1 for r in results if r['goal_reached'])}/{len(results)}\n\n")
            for r in results:
                status = "GOAL" if r["goal_reached"] else "FAIL"
                f.write(
                    f"Episode {r['episode']}: [{status}] "
                    f"reward={r['total_reward']:.2f}, "
                    f"steps={r['steps']}, max_x={r['max_x']}\n"
                )
        logger.info(f"Results saved to: {path}")
