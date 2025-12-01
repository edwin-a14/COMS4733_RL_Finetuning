"""PPO training script for RL finetuning of VLA policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict
import numpy as np

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from env.mujoco_env import FrankaPickPlaceEnv
from models.vla_dinov2 import VLADinoV2Config, VLADinoV2Policy
from rl.ppo_trainer import PPOTrainer, RolloutBuffer
from utils.config import load_config, save_config
from utils.logging import get_logger, setup_logging
from utils.seed import seed_everything


class ActionHistoryTracker:
    """Tracks action history for temporal context."""

    def __init__(self, history_length: int, action_dim: int, device: torch.device, action_stats: Dict[str, Any] = None):
        self.history_length = history_length
        self.action_dim = action_dim
        self.device = device
        self.action_stats = action_stats
        self.reset()

    def reset(self):
        """Reset history to zeros (or normalized zeros)."""
        if self.action_stats is not None:
            # Initialize with normalized zeros (matching dataset padding)
            # Dataset pads with raw zeros, then normalizes.
            # So we need (0 - mean) / std
            action_mean = torch.tensor(self.action_stats["mean"], dtype=torch.float32, device=self.device)
            action_std = torch.tensor(self.action_stats["std"], dtype=torch.float32, device=self.device)
            normalized_zero = (torch.zeros(self.action_dim, device=self.device) - action_mean) / (action_std + 1e-8)
            self.history = normalized_zero.unsqueeze(0).repeat(self.history_length, 1)
        else:
            self.history = torch.zeros(self.history_length, self.action_dim, device=self.device)

    def update(self, action: torch.Tensor):
        """Update history with new action."""
        # Shift history and add new action
        self.history = torch.cat([self.history[1:], action.unsqueeze(0)], dim=0)

    def get(self) -> torch.Tensor:
        """Get current history."""
        return self.history.clone()


def collect_rollout(
    env: FrankaPickPlaceEnv,
    policy: VLADinoV2Policy,
    buffer: RolloutBuffer,
    rollout_length: int,
    device: torch.device,
    action_std: float,
    action_stats: Dict[str, Any],
    instruction: str = "Pick up the red sphere and place it in the goal bin.",
    render: bool = False,
    uses_bce_gripper: bool = False,
) -> Dict[str, float]:
    """Collect a rollout using the current policy.

    Args:
        env: Environment to collect rollout in
        policy: Current policy
        buffer: Rollout buffer to store transitions
        rollout_length: Number of steps to collect
        device: Device to run policy on
        action_std: Standard deviation for action exploration
        action_stats: Action statistics for denormalization
        instruction: Language instruction for the task
        render: Whether to render the environment
        uses_bce_gripper: Whether the model uses BCE for gripper (requires thresholding)

    Returns:
        Dictionary of rollout metrics
    """
    policy.eval()

    # Action history tracker
    action_tracker = ActionHistoryTracker(
        history_length=policy.config.history_length,
        action_dim=policy.config.action_dim,
        device=device,
        action_stats=action_stats,
    )

    obs, info = env.reset()  # Environment returns (obs, info) tuple
    # FIX: Use instruction from environment (handles random colors)
    current_instruction = info.get("instruction", instruction)
    action_tracker.reset()

    episode_rewards = []
    episode_lengths = []
    current_episode_reward = 0
    current_episode_length = 0
    successes = []

    with torch.no_grad():
        for step in range(rollout_length):
            # Prepare observation
            rgb = torch.from_numpy(obs["rgb_static"]).to(device).float().permute(2, 0, 1).unsqueeze(0)

            # Normalize with ImageNet stats (matching evaluate_bc_mujoco.py)
            # rgb is [1, 3, 224, 224] in [0, 1]
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            rgb = (rgb - mean) / std

            # FIX: Mirror the image to match training distribution (Camera is mirrored)
            rgb = torch.flip(rgb, [3])

            proprio = torch.from_numpy(obs["proprio"]).to(device).float()

            # Normalize proprio to [-1, 1] using fixed joint limits (Franka Panda: ±2.8973 rad)
            joint_min = -2.8973
            joint_max = 2.8973
            proprio = 2.0 * (proprio - joint_min) / (joint_max - joint_min) - 1.0

            # Add timestep to proprio
            # FIX: Use current_episode_length for normalization to match evaluate_bc_mujoco.py
            # This ensures the policy sees consistent timestep features during training and eval
            timestep = torch.tensor([current_episode_length / 300.0], device=device, dtype=torch.float32)
            proprio = torch.cat([proprio, timestep], dim=-1).unsqueeze(0)

            # Get action history
            action_history = action_tracker.get().unsqueeze(0)

            # Get action, log prob, and value from policy
            action, log_prob, value = policy.get_action_and_value(
                rgb_static=rgb,
                instruction=[current_instruction],
                proprio=proprio,
                action_history=action_history,
                action_std=action_std,
            )

            # Denormalize action
            action_np = action.squeeze(0).cpu().numpy()
            if action_stats is not None:
                action_mean = np.array(action_stats["mean"])
                action_std_norm = np.array(action_stats["std"])
                
                if uses_bce_gripper:
                    # Only denormalize joints (0-6), leave gripper logit (7) alone
                    # Create a copy to avoid modifying the original action_np which might be needed for buffer?
                    # Actually buffer stores the raw network output (normalized/logit), so we can modify action_denorm
                    action_denorm = action_np.copy()
                    action_denorm[:7] = action_np[:7] * action_std_norm[:7] + action_mean[:7]
                else:
                    action_denorm = action_np * action_std_norm + action_mean
            else:
                action_denorm = action_np.copy()

            # Handle gripper based on training method
            if uses_bce_gripper:
                # Model trained with BCE: output is logit
                # Use CRISP BINARY threshold in logit space (more stable than sigmoid)
                gripper_logit = action_denorm[7]
                
                # Threshold tuning guide:
                # logit >  0.0  → prob > 0.50 (neutral, sigmoid threshold)
                # logit >  4.0  → prob > 0.98 (very confident open)
                
                GRIPPER_LOGIT_THRESHOLD = 4.0  # Matching evaluate_bc_mujoco.py
                
                # Crisp binary decision - no continuous values
                if gripper_logit > GRIPPER_LOGIT_THRESHOLD:
                    action_denorm[7] = 0.04  # Open
                else:
                    action_denorm[7] = 0.0   # Closed
            else:
                # Model trained with MSE: round to binary states if needed, or pass continuous
                # But for consistency with eval script:
                action_denorm[7] = 0.0 if action_denorm[7] < 0.02 else 0.04

            # Step environment (returns StepResult object, not tuple)
            step_result = env.step(action_denorm)
            next_obs = step_result.observation
            reward = step_result.reward
            done = step_result.terminated or step_result.truncated
            info = step_result.info

            if step % 20 == 0:
                dist = info.get("distance", -1.0)
                hist_grip = action_history[0, -1, 7].item()
                print(f"Step {step}: Gripper Logit={gripper_logit:.4f}, Action={action_denorm[7]:.4f}, Dist={dist:.4f}, HistGrip={hist_grip:.4f}")

            # Store in buffer (squeeze batch dimension)
            buffer.add(
                rgb=rgb.squeeze(0),
                instruction=current_instruction,
                proprio=proprio.squeeze(0),
                action_history=action_history.squeeze(0),
                action=action.squeeze(0),
                log_prob=log_prob.squeeze(0),
                value=value.squeeze(0),
                reward=reward,
                done=done,
            )

            # Update action history
            # FIX: If using BCE gripper, we must feed back the NORMALIZED EXECUTED action, not the logit
            action_for_history = action.squeeze(0).clone()
            if uses_bce_gripper and action_stats is not None:
                # Calculate normalized value of the executed binary gripper action
                # executed value is action_denorm[7] (0.0 or 0.04)
                # normalized = (executed - mean) / std
                gripper_val = action_denorm[7]
                gripper_mean = action_stats["mean"][7]
                gripper_std = action_stats["std"][7]
                gripper_normalized = (gripper_val - gripper_mean) / (gripper_std + 1e-8)
                action_for_history[7] = float(gripper_normalized)
            
            action_tracker.update(action_for_history)

            # Update episode metrics
            current_episode_reward += reward
            current_episode_length += 1

            if render:
                env.render()

            # Check if episode is done
            if done:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                successes.append(info.get("success", False))  # Fixed: use "success" not "is_success"

                # Reset environment and trackers
                obs, info = env.reset()  # Environment returns (obs, info) tuple
                # FIX: Update instruction for new episode
                current_instruction = info.get("instruction", instruction)
                action_tracker.reset()
                current_episode_reward = 0
                current_episode_length = 0
            else:
                obs = next_obs

    metrics = {
        "rollout/mean_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
        "rollout/mean_length": np.mean(episode_lengths) if episode_lengths else 0.0,
        "rollout/success_rate": np.mean(successes) if successes else 0.0,
        "rollout/num_episodes": len(episode_rewards),
    }

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VLA policy with PPO.")
    parser.add_argument("--config", type=str, default="rl/ppo_config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="BC checkpoint to load")
    parser.add_argument("--render", action="store_true", help="Render environment during rollouts")
    parser.add_argument("--quick-test", action="store_true", 
                        help="Run quick test with reduced epochs/rollouts (uses ppo_config_quick_test.yaml)")
    args = parser.parse_args()
    
    # Use quick test config if flag is set
    if args.quick_test:
        args.config = "rl/ppo_config_quick_test.yaml"
        print("=" * 60)
        print("QUICK TEST MODE ENABLED")
        print("=" * 60)
        print("Running reduced training for quick validation:")
        print("  - 3 epochs instead of 100")
        print("  - 256 steps/rollout instead of 2048")
        print("  - Expected runtime: ~5-10 minutes")
        print("=" * 60)
        print()

    # Load config
    config = load_config(args.config)
    policy_cfg = config["policy"]
    env_cfg = config["environment"]
    logging_cfg = config["logging"]

    setup_logging()
    logger = get_logger("train_ppo")

    # Set random seed
    seed = policy_cfg.get("seed", 0)
    seed_everything(seed)

    # Device setup - support M1/M2/M3 Mac GPU
    device_name = policy_cfg.get("device", "auto")
    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(logging_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup tensorboard
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    # Load action statistics
    action_stats_path = Path("dataset/action_stats.json")
    if action_stats_path.exists():
        with open(action_stats_path, "r") as f:
            action_stats = json.load(f)
        logger.info("Loaded action statistics for denormalization")
    else:
        action_stats = None
        logger.warning("No action statistics found - actions will not be denormalized")

    # Load BC checkpoint
    checkpoint_path = args.checkpoint or policy_cfg["checkpoint"]
    logger.info(f"Loading BC checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if model was trained with BCE for gripper
    uses_bce_gripper = checkpoint.get("uses_bce_gripper", False)
    if uses_bce_gripper:
        logger.info("Model trained with BCE for gripper - will apply thresholding to gripper output.")

    # Create model
    model_config = VLADinoV2Config(**checkpoint["config"])
    policy = VLADinoV2Policy(model_config)

    # Load BC weights (actor head)
    state_dict = checkpoint["model_state"]
    # if "proprio_projection.0.weight" in state_dict:
    #     # Slice to keep only first 7 dimensions (remove timestep)
    #     state_dict["proprio_projection.0.weight"] = state_dict["proprio_projection.0.weight"][:, :7]

    # Load BC weights (actor head)
    policy.load_state_dict(state_dict, strict=False)
    logger.info("Loaded BC weights (value_head initialized randomly)")

    # FREEZE BACKBONE: Only train the heads to prevent value function from destroying BC features
    for name, param in policy.named_parameters():
        if "head" in name:  # Train policy.head and policy.value_head
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # Verify trainable parameters
    trainable_params = [n for n, p in policy.named_parameters() if p.requires_grad]
    logger.info(f"Trainable parameters: {trainable_params}")

    policy.to(device)
    policy.train()

    # Create environment
    logger.info("Creating environment")
    env = FrankaPickPlaceEnv(
        asset_root=env_cfg.get("asset_root", "./env/mujoco_assets"),
        gui=args.render,  # Use gui parameter instead of render_mode
        seed=policy_cfg.get("seed", 42),
        reward_type=env_cfg.get("reward_type", "dense"),  # Now configurable!
    )
    logger.info(f"Using reward type: {env.reward_type}")
    # Note: max_steps is hardcoded to 340 in the environment
    env.max_steps = 600  # Increase max_steps to allow for slower policies (matching evaluate_bc_mujoco.py)

    # Create optimizer (convert to float in case YAML parses scientific notation as string)
    learning_rate = float(policy_cfg.get("learning_rate", 5e-6))
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    # Create PPO trainer
    ppo_trainer = PPOTrainer(
        policy=policy,
        optimizer=optimizer,
        clip_range=float(policy_cfg.get("clip_range", 0.2)),
        value_coef=float(policy_cfg.get("value_coef", 0.5)),
        entropy_coef=float(policy_cfg.get("entropy_coef", 0.01)),
        max_grad_norm=float(policy_cfg.get("max_grad_norm", 0.5)),
        action_std=float(policy_cfg.get("action_std", 0.1)),
        target_kl=float(policy_cfg.get("target_kl", 0.01)),
    )

    # Create rollout buffer
    rollout_buffer = RolloutBuffer(
        buffer_size=int(policy_cfg.get("rollout_length", 2048)),
        action_dim=model_config.action_dim,
        history_length=model_config.history_length,
        device=device,
    )

    # Training loop (ensure all numeric types are correct)
    num_epochs = int(policy_cfg.get("num_epochs", 10))
    rollout_length = int(policy_cfg.get("rollout_length", 2048))
    ppo_epochs = int(policy_cfg.get("ppo_epochs", 4))
    batch_size = int(policy_cfg.get("batch_size", 64))
    action_std = float(policy_cfg.get("action_std", 0.1))
    gamma = float(policy_cfg.get("gamma", 0.99))
    gae_lambda = float(policy_cfg.get("gae_lambda", 0.95))

    logger.info("Starting PPO training")
    logger.info(f"Epochs: {num_epochs}, Rollout length: {rollout_length}")

    global_step = 0
    best_success_rate = -1.0  # Track best success rate
    best_mean_reward = -float('inf')  # Track best reward for tie-breaking
    best_checkpoint_path = output_dir / "ppo_best.pt"

    for epoch in range(num_epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"{'='*50}")

        # Collect rollouts
        logger.info("Collecting rollouts...")
        rollout_buffer.reset()

        rollout_metrics = collect_rollout(
            env=env,
            policy=policy,
            buffer=rollout_buffer,
            rollout_length=rollout_length,
            device=device,
            action_std=action_std,
            action_stats=action_stats,
            render=args.render,
            uses_bce_gripper=uses_bce_gripper,
        )

        # Get last value for GAE
        # Use the last observation from the buffer
        with torch.no_grad():
            last_rgb = rollout_buffer.rgb_buffer[-1].unsqueeze(0).to(device)
            last_proprio = rollout_buffer.proprio_buffer[-1].unsqueeze(0).to(device)
            last_action_history = rollout_buffer.action_history_buffer[-1].unsqueeze(0).to(device)
            last_instruction = [rollout_buffer.instruction_buffer[-1]]

            last_value = policy.get_value(
                rgb_static=last_rgb,
                instruction=last_instruction,
                proprio=last_proprio,
                action_history=last_action_history,
            )

        # Compute returns and advantages
        advantages, returns = rollout_buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        # Get batch for training
        batch = rollout_buffer.get(advantages, returns)

        # Train with PPO
        logger.info("Training with PPO...")
        # CRITICAL: Use eval mode during PPO update to disable dropout!
        # If we use train(), dropout will cause the policy to output different values
        # than during rollout (where eval() was used), causing massive KL divergence immediately.
        policy.eval() 
        train_metrics = ppo_trainer.train_step(
            batch=batch,
            num_epochs=ppo_epochs,
            batch_size=batch_size,
        )

        # Log metrics
        global_step += rollout_length

        for key, value in rollout_metrics.items():
            writer.add_scalar(key, value, global_step)
            logger.info(f"  {key}: {value:.4f}")

        for key, value in train_metrics.items():
            writer.add_scalar(key, value, global_step)
            logger.info(f"  {key}: {value:.4f}")

        # Check for divergence (NaN or extreme values)
        if np.isnan(train_metrics["loss/total"]) or np.isinf(train_metrics["loss/total"]):
            logger.error(f"Training diverged! Loss is {train_metrics['loss/total']}")
            logger.error("Stopping training and saving checkpoint...")
            checkpoint_path = output_dir / f"ppo_diverged_epoch_{epoch + 1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state": policy.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": model_config.__dict__,
                "global_step": global_step,
            }, checkpoint_path)
            break

        # Check for extreme losses (potential divergence)
        if train_metrics["loss/total"] > 1000:
            logger.warning(f"Very high loss detected: {train_metrics['loss/total']:.2f}")
            logger.warning("Training may be diverging. Consider lowering learning rate.")

        # Save best checkpoint based on success rate (with reward tie-breaking)
        current_success_rate = rollout_metrics.get("rollout/success_rate", 0.0)
        current_mean_reward = rollout_metrics.get("rollout/mean_reward", -float('inf'))
        
        is_best = False
        if current_success_rate > best_success_rate:
            is_best = True
        elif current_success_rate == best_success_rate and current_mean_reward > best_mean_reward:
            is_best = True
            
        if is_best:
            best_success_rate = current_success_rate
            best_mean_reward = current_mean_reward
            logger.info(f"Saving best checkpoint with uses_bce_gripper={uses_bce_gripper}, action_stats={'present' if action_stats else 'None'}")
            torch.save({
                "epoch": epoch + 1,
                "model_state": policy.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": model_config.__dict__,
                "global_step": global_step,
                "success_rate": best_success_rate,
                "mean_reward": best_mean_reward,
                "action_stats": action_stats,
                "uses_bce_gripper": uses_bce_gripper,
            }, best_checkpoint_path)
            logger.info(f"✓ New best model: Success={best_success_rate:.2%}, Reward={best_mean_reward:.2f} - Saved to {best_checkpoint_path}")

    # Save final checkpoint (last epoch)
    final_checkpoint_path = output_dir / "ppo_last.pt"
    torch.save({
        "epoch": num_epochs,
        "model_state": policy.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": model_config.__dict__,
        "global_step": global_step,
        "action_stats": action_stats,
        "uses_bce_gripper": uses_bce_gripper,
    }, final_checkpoint_path)
    logger.info(f"Saved final checkpoint to {final_checkpoint_path}")
    logger.info(f"Best success rate achieved: {best_success_rate:.2%} (saved at {best_checkpoint_path})")

    writer.close()
    env.close()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
