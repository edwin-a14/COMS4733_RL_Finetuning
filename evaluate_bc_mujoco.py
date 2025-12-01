"""Evaluate a trained behavioral cloning policy in MuJoCo."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor
import torchvision.transforms as T

from env.mujoco_env import FrankaPickPlaceEnv
from models.vla_dinov2 import VLADinoV2Config, VLADinoV2Policy
from utils.config import load_config
from utils.logging import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a behavioral cloning policy in MuJoCo")
    parser.add_argument("--config", type=str, default="configs/openvla_dinov2_bc.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes-static", type=int, default=None,
                        help="Number of static episodes to run when evaluating both modes.")
    parser.add_argument("--episodes-hindered", type=int, default=None,
                        help="Number of hindered episodes to run when evaluating both modes.")
    parser.add_argument(
        "--mode",
        choices=["static", "hindered"],
        default=None,
        help="If provided, evaluate only the selected mode (static or hindered).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of episodes to run when --mode is specified.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable GUI rendering for faster evaluation.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save video of the first successful episode for each mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    logger = get_logger("evaluate_bc")

    config = load_config(args.config)
    evaluation_cfg = config.get("evaluation", {})
    model_cfg = config.get("model", {})

    episodes_static = args.episodes_static or evaluation_cfg.get("episodes_static", 50)
    episodes_hindered = args.episodes_hindered or evaluation_cfg.get("episodes_hindered", 50)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "config" in checkpoint:
        model_cfg = checkpoint["config"]
    
    # Load action statistics for denormalization
    action_stats = checkpoint.get("action_stats", None)
    if action_stats is None:
        # Fallback: Try to load from dataset/action_stats.json
        stats_path = Path("dataset/action_stats.json")
        if stats_path.exists():
            import json
            with open(stats_path, "r") as f:
                action_stats = json.load(f)
            logger.info(f"Loaded action statistics from {stats_path} (fallback).")
        else:
            logger.warning("Action statistics not found in checkpoint or disk. Actions will not be denormalized.")
    else:
        logger.info("Loaded action statistics from checkpoint for denormalization.")
    
    # Check if model was trained with BCE (using logits) for gripper
    uses_bce_gripper = checkpoint.get("uses_bce_gripper", False)
    if uses_bce_gripper:
        logger.info("Model trained with BCE for gripper - will apply sigmoid to gripper output.")
    else:
        # DEBUG: Force True for PPO checkpoints that missed the flag
        logger.warning("uses_bce_gripper not found, forcing TRUE for debugging PPO checkpoint.")
        uses_bce_gripper = True

    model_config = VLADinoV2Config(**model_cfg)
    policy = VLADinoV2Policy(model_config)
    # Use strict=False to handle BC checkpoints that don't have value_head (RL critic)
    policy.load_state_dict(checkpoint["model_state"], strict=False)
    logger.info("Loaded BC checkpoint (value_head not present, only actor/policy head)")
    policy.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)

    # Use fast processor and disable rescaling (images already in [0,1] range from environment)
    image_processor = AutoImageProcessor.from_pretrained(model_config.vision_encoder, use_fast=True, do_rescale=False)

    asset_root = evaluation_cfg.get("asset_root", "./env/mujoco_assets")
    # Disable rendering if --no-render flag is set, otherwise use config setting
    enable_gui = evaluation_cfg.get("render", False) and not args.no_render
    env = FrankaPickPlaceEnv(
        asset_root=asset_root,
        gui=enable_gui,
        seed=evaluation_cfg.get("seed", 0),
    )
    # Increase max_steps to allow for slower policies
    env.max_steps = 600

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.mode is not None:
            selected_mode = args.mode
            default_episodes = episodes_static if selected_mode == "static" else episodes_hindered
            episodes = args.episodes or default_episodes
            if episodes is None:
                raise ValueError("Number of episodes must be specified via --episodes or the evaluation config.")
            results, success_rate = run_rollouts(
                env,
                policy,
                image_processor,
                device,
                episodes,
                mode=selected_mode,
                action_stats=action_stats,
                uses_bce_gripper=uses_bce_gripper,
                save_video=args.save_video,
            )
            save_results(results_dir / f"{selected_mode}_eval.csv", results)
            if selected_mode == "static":
                save_plot(results_dir / "comparison_plot.png", success_rate, None)
            
            avg_reward = sum(item["reward"] for item in results) / len(results) if results else 0.0
            logger.info(
                "%s success rate: %.2f%% | Avg Reward: %.4f",
                selected_mode.capitalize(),
                success_rate * 100 if success_rate is not None else float("nan"),
                avg_reward
            )
            return

        static_results, static_rate = run_rollouts(
            env,
            policy,
            image_processor,
            device,
            episodes_static,
            mode="static",
            action_stats=action_stats,
            uses_bce_gripper=uses_bce_gripper,
            save_video=args.save_video,
        )
        hindered_results, hindered_rate = run_rollouts(
            env,
            policy,
            image_processor,
            device,
            episodes_hindered,
            mode="hindered",
            action_stats=action_stats,
            uses_bce_gripper=uses_bce_gripper,
            save_video=args.save_video,
        )

        save_results(results_dir / "static_eval.csv", static_results)
        save_results(results_dir / "hindered_eval.csv", hindered_results)
        save_plot(results_dir / "comparison_plot.png", static_rate, hindered_rate)

        logger.info("Static success rate: %.2f%%", static_rate * 100 if static_rate is not None else float("nan"))
        logger.info("Hindered success rate: %.2f%%", hindered_rate * 100 if hindered_rate is not None else float("nan"))
        if static_rate is not None and hindered_rate is not None:
            logger.info("Performance gap (static - hindered): %.2f%%", (static_rate - hindered_rate) * 100)
    finally:
        env.close()


def run_rollouts(
    env: FrankaPickPlaceEnv,
    policy: VLADinoV2Policy,
    image_processor,
    device: torch.device,
    episodes: int,
    mode: str,
    action_stats: Dict[str, List[float]] | None = None,
    uses_bce_gripper: bool = False,
    save_video: bool = False,
) -> Tuple[List[Dict[str, float]], float | None]:
    logger = get_logger("evaluate_bc")
    results: List[Dict[str, float]] = []
    video_saved = False

    for episode in range(episodes):
        observation, info = env.reset(hindered=mode == "hindered")
        done = False
        total_reward = 0.0
        success = False
        step = 0
        frames = []
        # Use the actual instruction from environment (e.g., "Pick up the blue sphere...")
        instruction_text = info.get("instruction", "Pick up the sphere and place it in the goal bin.")
        
        # Initialize action history buffer for closed-loop control
        history_length = policy.config.history_length
        action_dim = policy.config.action_dim
        
        # Initialize with normalized zeros (matching dataset padding)
        # Dataset pads with raw zeros, then normalizes.
        # So we need (0 - mean) / std
        if action_stats is not None:
            action_mean = torch.tensor(action_stats["mean"], dtype=torch.float32, device=device)
            action_std = torch.tensor(action_stats["std"], dtype=torch.float32, device=device)
            normalized_zero = (torch.zeros(action_dim, device=device) - action_mean) / (action_std + 1e-8)
            action_history = normalized_zero.unsqueeze(0).repeat(history_length, 1)
        else:
            action_history = torch.zeros((history_length, action_dim), dtype=torch.float32, device=device)

        while not done:
            step += 1
            
            # Capture frame for video
            if save_video and not video_saved:
                frame = env.render(mode="rgb_array")
                frame = (frame * 255).astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frames.append(frame)

            # Pass timestep for temporal awareness (helps model learn "close at ~step 120, open at ~step 320")
            rgb_tensor, proprio_tensor = preprocess_observation(observation, image_processor, device, timestep=step, max_steps=300)
            
            # DEBUG: Save the first frame of the first episode
            # if episode == 0 and step == 1:
            #     import torchvision.utils as vutils
            #     # Denormalize for visualization
            #     mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            #     std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            #     vis_tensor = rgb_tensor * std + mean
            #     vutils.save_image(vis_tensor, "debug_input_image.png")
            #     logger.info("Saved debug_input_image.png")

            with torch.no_grad():
                instructions = [instruction_text] * rgb_tensor.size(0)
                # Pass action history for closed-loop control
                action = policy(
                    rgb_static=rgb_tensor, 
                    proprio=proprio_tensor, 
                    instruction=instructions,
                    action_history=action_history.unsqueeze(0)  # Add batch dimension: (1, H, D)
                ).cpu().numpy()
            
            # Denormalize action if statistics are available
            if action_stats is not None:
                action_mean = np.array(action_stats["mean"], dtype=np.float32)
                action_std = np.array(action_stats["std"], dtype=np.float32)
                
                if uses_bce_gripper:
                    # Only denormalize joints (0-6), leave gripper logit (7) alone
                    action[0, :7] = action[0, :7] * action_std[:7] + action_mean[:7]
                else:
                    # Denormalize ALL dimensions including gripper (MSE training)
                    action[0] = action[0] * action_std + action_mean
            
            # Handle gripper based on training method
            if uses_bce_gripper:
                # Model trained with BCE: output is logit
                # Use CRISP BINARY threshold in logit space (more stable than sigmoid)
                gripper_logit = action[0, 7]
                
                # Threshold tuning guide:
                # logit >  0.0  → prob > 0.50 (neutral, sigmoid threshold)
                # logit > -0.4  → prob > 0.40 (easier to open, conservative)
                # logit > -0.8  → prob > 0.31 (very easy to open)
                # logit >  0.5  → prob > 0.62 (harder to open, more selective)
                
                GRIPPER_LOGIT_THRESHOLD = 4.0  # Adjusted based on calibration (model outputs ~3.5 when grasping)
                
                # Crisp binary decision - no continuous values
                if gripper_logit > GRIPPER_LOGIT_THRESHOLD:
                    action[0, 7] = 0.04  # Open
                else:
                    action[0, 7] = 0.0   # Closed
            else:
                # Model trained with MSE: round to binary states
                action[0, 7] = 0.0 if action[0, 7] < 0.02 else 0.04
            
            # Removed verbose step-by-step logging for faster evaluation
            # (Was logging every 20 steps, now only episode-level results)
            
            step_result = env.step(action.squeeze(0))
            observation = step_result.observation
            total_reward += float(step_result.reward)
            done = bool(step_result.terminated or step_result.truncated)
            success = bool(step_result.info.get("success", success))
            
            # Update action history buffer (shift and append for next timestep)
            # This enables closed-loop control by letting model see its recent actions
            action_normalized = torch.from_numpy(action[0]).float().to(device)
            if action_stats is not None:
                # Re-normalize the action for history (model expects normalized actions)
                action_mean_tensor = torch.from_numpy(action_mean).float().to(device)
                action_std_tensor = torch.from_numpy(action_std).float().to(device)
                action_normalized = (action_normalized - action_mean_tensor) / (action_std_tensor + 1e-8)
            action_history = torch.cat([action_history[1:], action_normalized.unsqueeze(0)], dim=0)
            
            # Sync viewer for GUI visualization (if enabled)
            if hasattr(env, 'viewer') and env.viewer is not None:
                env.viewer.sync()
            
            # DEBUG: Save trajectory for ALL episodes
            # with open("debug_trajectory.csv", "a") as f:
            #     # Format: episode, step, gripper_logit, gripper_action, reward, distance, obj_x,obj_y,obj_z,ee_x,ee_y,ee_z
            #     dist = step_result.info.get("distance", 0.0)
                
            #     # Get positions
            #     try:
            #         target_site_id = env._object_site_ids[env.target_color]
            #         obj_pos = env.data.site_xpos[target_site_id]
            #         ee_pos = env.data.site_xpos[env._gripper_site_id]
            #         f.write(f"{episode},{step},{gripper_logit},{action[0, 7]},{step_result.reward},{dist},{obj_pos[0]},{obj_pos[1]},{obj_pos[2]},{ee_pos[0]},{ee_pos[1]},{ee_pos[2]}\n")
            #     except Exception as e:
            #         print(f"Logging failed: {e}")
            #         f.write(f"{episode},{step},{gripper_logit},{action[0, 7]},{step_result.reward},{dist},0,0,0,0,0,0\n")

        results.append({"episode": episode, "success": float(success), "reward": total_reward})

        # Save video if successful or if it's the first episode (for debugging)
        if (success or episode == 0) and save_video and frames:
            status = "success" if success else "failed"
            video_path = Path(f"results/video_{mode}_{status}_ep{episode}.mp4")
            height, width, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, 25.0, (width, height))
            for frame in frames:
                out.write(frame)
            out.release()
            logger.info(f"Saved video of {status} {mode} episode {episode} to {video_path}")

    if not results:
        return results, 0.0

    success_rate = sum(item["success"] for item in results) / len(results)
    return results, success_rate


def preprocess_observation(observation, image_processor, device: torch.device, timestep: int = 0, max_steps: int = 300) -> Tuple[torch.Tensor, torch.Tensor]:
    """Preprocess observation for model input.
    
    Args:
        observation: Dict with 'rgb_static' and 'proprio' keys
        image_processor: HuggingFace image processor
        device: torch device
        timestep: Current timestep in episode (for temporal awareness)
        max_steps: Maximum episode length for timestep normalization (must match training: 300)
    
    Returns:
        rgb_tensor: Preprocessed RGB image
        proprio_tensor: Proprio + timestep concatenated (8-dim)
    """
    if isinstance(observation, dict):
        # Use explicit key checking to avoid numpy array ambiguity with 'or' operator
        rgb = observation.get("rgb_static") if "rgb_static" in observation else observation.get("image")
        proprio = observation.get("proprio") if "proprio" in observation else observation.get("robot_qpos")
    else:
        raise ValueError("Observation must be a dictionary containing rgb_static/image and proprio data")

    if rgb is None:
        raise ValueError("Observation missing rgb information")

    if isinstance(rgb, torch.Tensor):
        rgb_tensor = rgb.detach().cpu()
        if rgb_tensor.ndim == 3 and rgb_tensor.shape[0] in {1, 3}:
            rgb_array = rgb_tensor.permute(1, 2, 0).numpy()
        else:
            raise ValueError("Unsupported rgb tensor shape")
    else:
        if isinstance(rgb, np.ndarray) and rgb.ndim == 3:
            rgb_array = rgb.astype(np.float32)
        else:
            raise ValueError("Unsupported rgb format")
        if rgb_array.max() > 1.0:
            rgb_array = rgb_array / 255.0

    # Manual preprocessing to match training exactly
    # 1. Resize to 224x224
    # 2. Normalize with ImageNet stats
    
    # Convert to tensor (C, H, W) and normalize to [0, 1]
    rgb_tensor = torch.from_numpy(rgb_array).permute(2, 0, 1).float()
    
    # Resize
    resize = T.Resize((224, 224))
    rgb_tensor = resize(rgb_tensor)
    
    # Normalize
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    rgb_tensor = normalize(rgb_tensor).unsqueeze(0).to(device)

    # FIX: Mirror the image to match training distribution (Camera is mirrored)
    rgb_tensor = torch.flip(rgb_tensor, [3])

    if proprio is None:
        proprio_tensor = torch.zeros(1, 8, device=device)  # 7 joints + 1 timestep
    else:
        if isinstance(proprio, torch.Tensor):
            proprio_tensor = proprio.float().to(device).unsqueeze(0)
        else:
            proprio_array = np.asarray(proprio, dtype=np.float32)
            proprio_tensor = torch.from_numpy(proprio_array).float().to(device).unsqueeze(0)
        if proprio_tensor.ndim == 1:
            proprio_tensor = proprio_tensor.unsqueeze(0)
        
        # Normalize proprio to [-1, 1] using fixed joint limits (Franka Panda: ±2.8973 rad)
        # This ensures consistent normalization across all observations
        joint_min = -2.8973
        joint_max = 2.8973
        proprio_tensor = 2.0 * (proprio_tensor - joint_min) / (joint_max - joint_min) - 1.0
        
        # TEST: Zero out proprioception to force reliance on vision
        # proprio_tensor = torch.zeros_like(proprio_tensor)
    
    # Add normalized timestep as 8th dimension (for temporal awareness)
    timestep_normalized = timestep / max(max_steps, 1)  # Normalize to [0, 1]
    timestep_tensor = torch.tensor([[timestep_normalized]], dtype=torch.float32, device=device)
    
    # Concatenate: (1, 7) + (1, 1) → (1, 8)
    proprio_tensor = torch.cat([proprio_tensor, timestep_tensor], dim=-1)

    return rgb_tensor, proprio_tensor


def _normalize_range(tensor: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Normalize tensor to [low, high] range to match training preprocessing."""
    minimum = tensor.min()
    maximum = tensor.max()
    if maximum == minimum:
        return torch.zeros_like(tensor)
    scaled = (tensor - minimum) / (maximum - minimum)
    return scaled * (high - low) + low


def save_results(path: Path, rows: Iterable[Dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode", "success", "reward"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_plot(path: Path, static_rate: float | None, hindered_rate: float | None) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib optional
        get_logger("evaluate_bc").warning("matplotlib unavailable: %s", exc)
        return

    rates = []
    labels = []
    if static_rate is not None:
        labels.append("Static")
        rates.append(static_rate)
    if hindered_rate is not None:
        labels.append("Hindered")
        rates.append(hindered_rate)

    if not rates:
        return

    plt.figure(figsize=(4, 4))
    plt.bar(labels, rates, color=["#4CAF50", "#FF5722"][: len(rates)])
    plt.ylim(0, 1)
    plt.ylabel("Success Rate")
    plt.title("BC Policy Evaluation")
    for idx, rate in enumerate(rates):
        plt.text(idx, rate + 0.02, f"{rate*100:.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    main()
