"""Evaluation script for PPO-trained VLA policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np

import torch
import torchvision.transforms as T
from env.mujoco_env import FrankaPickPlaceEnv
from models.vla_dinov2 import VLADinoV2Config, VLADinoV2Policy
from utils.logging import get_logger, setup_logging


class ActionHistoryTracker:
    """Tracks action history for temporal context."""

    def __init__(self, history_length: int, action_dim: int, device: torch.device):
        self.history_length = history_length
        self.action_dim = action_dim
        self.device = device
        self.reset()

    def reset(self):
        """Reset history to zeros."""
        self.history = torch.zeros(self.history_length, self.action_dim, device=self.device)

    def update(self, action: torch.Tensor):
        """Update history with new action."""
        self.history = torch.cat([self.history[1:], action.unsqueeze(0)], dim=0)

    def get(self) -> torch.Tensor:
        """Get current history."""
        return self.history.clone()


def evaluate_policy(
    env: FrankaPickPlaceEnv,
    policy: VLADinoV2Policy,
    num_episodes: int,
    device: torch.device,
    action_stats: Dict[str, Any] | None = None,
    instruction: str = "Pick up the red sphere and place it in the goal bin.",
    render: bool = False,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """Evaluate policy on multiple episodes.

    Args:
        env: Environment to evaluate in
        policy: Policy to evaluate
        num_episodes: Number of episodes to run
        device: Device to run policy on
        action_stats: Action statistics for denormalization
        instruction: Language instruction for the task
        render: Whether to render the environment
        deterministic: Whether to use deterministic (mean) actions

    Returns:
        Dictionary of evaluation metrics
    """
    policy.eval()

    # Action history tracker
    action_tracker = ActionHistoryTracker(
        history_length=policy.config.history_length,
        action_dim=policy.config.action_dim,
        device=device,
    )

    episode_rewards = []
    episode_lengths = []
    successes = []

    with torch.no_grad():
        for episode in range(num_episodes):
            obs, info = env.reset()  # Environment returns (obs, info) tuple
            action_tracker.reset()

            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                # Prepare observation
                # (H, W, C) -> (C, H, W)
                rgb = torch.from_numpy(obs["rgb_static"]).to(device).float().permute(2, 0, 1)
                
                # Resize to 224x224
                resize = T.Resize((224, 224))
                rgb = resize(rgb)
                
                # Normalize with ImageNet stats
                normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                rgb = normalize(rgb).unsqueeze(0)

                proprio = torch.from_numpy(obs["proprio"]).to(device).float()

                # Add timestep to proprio
                timestep = torch.tensor([episode_length / env.max_steps], device=device, dtype=torch.float32)
                proprio = torch.cat([proprio, timestep], dim=-1).unsqueeze(0)

                # Get action history
                action_history = action_tracker.get().unsqueeze(0)

                # Get action from policy
                if deterministic:
                    # Use mean action (no sampling)
                    action = policy(
                        rgb_static=rgb,
                        instruction=[instruction],
                        proprio=proprio,
                        action_history=action_history,
                    )
                else:
                    # Sample action
                    action, _, _ = policy.get_action_and_value(
                        rgb_static=rgb,
                        instruction=[instruction],
                        proprio=proprio,
                        action_history=action_history,
                        action_std=0.1,
                    )

                # Denormalize action
                action_np = action.squeeze(0).cpu().numpy()
                if action_stats is not None:
                    action_mean = np.array(action_stats["mean"])
                    action_std_norm = np.array(action_stats["std"])
                    action_denorm = action_np * action_std_norm + action_mean
                else:
                    action_denorm = action_np

                # Step environment (returns StepResult object, not tuple)
                step_result = env.step(action_denorm)
                obs = step_result.observation
                reward = step_result.reward
                done = step_result.terminated or step_result.truncated
                info = step_result.info

                # Update action history
                action_tracker.update(action.squeeze(0))

                # Update metrics
                episode_reward += reward
                episode_length += 1

                if render:
                    env.render()

                # Safety check for max steps
                if episode_length >= env.max_steps:
                    done = True

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            successes.append(info.get("success", False))  # Fixed: use "success" not "is_success"

            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Reward={episode_reward:.2f}, "
                  f"Length={episode_length}, "
                  f"Success={info.get('is_success', False)}")

    metrics = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "success_rate": np.mean(successes),
        "num_episodes": num_episodes,
    }

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPO-trained VLA policy.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to PPO checkpoint")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument("--render", action="store_true", help="Render environment during evaluation")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions (mean)")
    parser.add_argument("--output", type=str, default=None, help="Path to save evaluation results")
    args = parser.parse_args()

    setup_logging()
    logger = get_logger("evaluate_ppo")

    # Device setup - support M1/M2/M3 Mac GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Create model
    model_config = VLADinoV2Config(**checkpoint["config"])
    policy = VLADinoV2Policy(model_config)
    policy.load_state_dict(checkpoint["model_state"])
    policy.to(device)
    policy.eval()

    logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")

    # Load action statistics
    action_stats_path = Path("dataset/action_stats.json")
    if action_stats_path.exists():
        with open(action_stats_path, "r") as f:
            action_stats = json.load(f)
        logger.info("Loaded action statistics for denormalization")
    else:
        action_stats = None
        logger.warning("No action statistics found - actions will not be denormalized")

    # Create environment
    logger.info("Creating environment")
    env = FrankaPickPlaceEnv(
        asset_root="./env/mujoco_assets",
        gui=args.render,  # Use gui parameter
        seed=42,
        reward_type="dense",  # Can be made configurable via args if needed
    )

    # Evaluate policy
    logger.info(f"Evaluating policy for {args.num_episodes} episodes")
    logger.info(f"Deterministic: {args.deterministic}")

    metrics = evaluate_policy(
        env=env,
        policy=policy,
        num_episodes=args.num_episodes,
        device=device,
        action_stats=action_stats,
        render=args.render,
        deterministic=args.deterministic,
    )

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"Mean Length: {metrics['mean_length']:.1f} ± {metrics['std_length']:.1f}")
    print(f"Success Rate: {metrics['success_rate']:.2%}")
    print(f"Number of Episodes: {metrics['num_episodes']}")
    print("="*50)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved results to {output_path}")

    env.close()
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
