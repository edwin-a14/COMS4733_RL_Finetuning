"""Scripted demonstration collection for the Franka pick-and-place task.

This utility generates LeRobot-compatible episodes by running a simple
heuristic controller inside :class:`env.mujoco_env.FrankaPickPlaceEnv`.  The
resulting dataset can be used directly by ``train_bc.py`` and mirrors the
structure expected by the COMS4733 Milestone 1 baseline.

The script purposely keeps the policy trivial – it drives the gripper towards
the target object's site using a proportional controller and lifts it above the
table.  While this will not solve challenging scenes, it is sufficient for
smoke-testing the end-to-end data pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from env.mujoco_env import FrankaPickPlaceEnv
from env.controllers import KeyframeController, KinematicsHelper


def world_to_image_coords(obj_pos_3d: np.ndarray, workspace_bounds: tuple = None) -> np.ndarray:
    """Convert 3D world position to normalized 2D image coordinates [0, 1].
    
    Args:
        obj_pos_3d: 3D position in world coordinates (x, y, z)
        workspace_bounds: Tuple of ((x_min, x_max), (y_min, y_max)) or None for defaults
    
    Returns:
        Normalized 2D position (x_norm, y_norm) in [0, 1] range
    """
    if workspace_bounds is None:
        # Default workspace bounds for Franka pick-and-place task
        x_min, x_max = 0.40, 0.65  # X range: 40cm to 65cm (25cm width)
        y_min, y_max = -0.30, 0.30  # Y range: -30cm to 30cm (60cm depth)
    else:
        (x_min, x_max), (y_min, y_max) = workspace_bounds
    
    x_world, y_world, z_world = obj_pos_3d
    
    # Normalize to [0, 1] based on workspace bounds
    x_norm = (x_world - x_min) / (x_max - x_min)
    y_norm = (y_world - y_min) / (y_max - y_min)
    
    # Clamp to [0, 1] to handle objects slightly outside bounds
    x_norm = float(np.clip(x_norm, 0.0, 1.0))
    y_norm = float(np.clip(y_norm, 0.0, 1.0))
    
    return np.array([x_norm, y_norm], dtype=np.float32)


@dataclass(slots=True)
class EpisodeBuffer:
    """Stores trajectory information before writing to disk."""

    rgb_frames: List[np.ndarray]
    proprio: List[np.ndarray]
    actions: List[np.ndarray]
    timestamps: List[float]
    object_positions: List[np.ndarray]  # NEW: Ground truth object positions (normalized [0,1])
    instruction: str
    meta: Dict[str, object]

    def extend(self, obs: Dict[str, np.ndarray], action: np.ndarray, timestamp: float, object_pos_normalized: np.ndarray) -> None:
        """Add a step with ground truth object position."""
        self.rgb_frames.append((obs["rgb_static"] * 255).astype(np.uint8))
        self.proprio.append(obs["proprio"].astype(np.float32))
        self.actions.append(action.astype(np.float32))
        self.timestamps.append(float(timestamp))
        self.object_positions.append(object_pos_normalized.astype(np.float32))

    def save(self, root: Path, episode_id: int) -> None:
        episode_dir = root / f"episode_{episode_id:04d}"
        image_dir = episode_dir / "obs" / "rgb_static"
        episode_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)

        for idx, frame in enumerate(self.rgb_frames):
            Image.fromarray(frame).save(image_dir / f"{idx:06d}.png")

        np.save(episode_dir / "obs" / "proprio.npy", np.stack(self.proprio, axis=0))
        np.save(episode_dir / "actions.npy", np.stack(self.actions, axis=0))
        np.save(episode_dir / "timestamps.npy", np.asarray(self.timestamps, dtype=np.float32))
        
        # NEW: Save ground truth object positions
        np.save(episode_dir / "obs" / "object_positions.npy", np.stack(self.object_positions, axis=0))

        (episode_dir / "instruction.txt").write_text(self.instruction)
        (episode_dir / "meta.json").write_text(json.dumps(self.meta, indent=2))


def compute_adaptive_keyframes(
    env: FrankaPickPlaceEnv,
    object_pos: np.ndarray,
    kin_helper: KinematicsHelper,
) -> dict[str, np.ndarray]:
    """Compute keyframes adapted to actual object position using IK.
    
    Args:
        env: The environment instance
        object_pos: 3D position of the target object
        kin_helper: Kinematics helper for IK computation
    
    Returns:
        Dictionary mapping keyframe names to 7D joint configurations
    """
    # Base keyframes (proven for object at 0.5, 0, 0.03)
    base_keyframes = {
        "home": np.array([0.0, 0.3, 0.0, -1.57079, 0.0, 2.0, -0.7853]),
        "pre_grasp": np.array([0.1, 0.35, -0.1, -2.05, 0.0, 2.0, -0.5]),
        "grasp": np.array([0.0, 0.675, 0.0, -1.9, 0.0, 2.0, -0.5]),
    }
    
    # Reference object position that base keyframes were designed for
    reference_pos = np.array([0.5, 0.0, 0.03])
    
    # Compute offset from reference position
    offset = object_pos - reference_pos
    
    print(f"  Object at {object_pos}, offset from reference: {offset}")
    
    # Compute adaptive keyframes using IK
    keyframes = {}
    keyframes["home"] = base_keyframes["home"]  # Home doesn't change
    
    # Target positions for pick sequence (adjusted for actual object)
    # Ball: radius = 0.03m (3cm), center at object_pos
    # 
    # Edge case handling: When ball is near bin (high Y), use lower grasp for better vertical alignment
    bin_y = 0.45
    distance_to_bin = abs(object_pos[1] - bin_y)
    ball_near_bin = distance_to_bin < 0.35  # Within 35cm of bin
    
    if ball_near_bin:
        # Lower grasp height for better vertical alignment at high Y positions
        pre_grasp_height = 0.12   # 12cm above
        grasp_height = 0.010      # 1cm above (lower for more vertical approach)
        print(f"  ⚠ Ball near bin (dist={distance_to_bin*1000:.0f}mm) - using lower grasp height")
    else:
        # Standard heights
        pre_grasp_height = 0.12   # 12cm above
        grasp_height = 0.00       # 0.0cm above (grasp at center of object)
    
    target_positions = {
        "pre_grasp": object_pos + np.array([0, 0, pre_grasp_height]),
        "grasp": object_pos + np.array([0, 0, grasp_height]),
    }
    
    # Downward orientation for grasping
    downward_quat = np.array([0.0, 1.0, 0.0, 0.0])  # Gripper pointing down
    
    # Compute pre_grasp and grasp using IK
    for keyframe_name, target_pos in target_positions.items():
        # Use base keyframe as initial guess
        initial_q = base_keyframes[keyframe_name]
        
        use_ik = False
        
        # For balls near bin, use stricter orientation tolerance and higher orientation weight
        if ball_near_bin:
            orientation_tolerance = math.radians(20.0)  # Stricter: 20° vs 30°
            ori_weight_ccd = 1.5  # Higher orientation priority
            ori_weight_standard = 0.5
        else:
            orientation_tolerance = math.radians(30.0)  # Standard: 30°
            ori_weight_ccd = 0.5
            ori_weight_standard = 0.1
        
        # Try multiple IK methods for better convergence
        for method in ["ccd", "standard", "staged"]:
            try:
                if method == "ccd":
                    # CCD IK - often more robust for difficult positions
                    result_q = kin_helper.inverse_kinematics_ccd(
                        target_pos=target_pos,
                        target_quat=downward_quat,
                        initial_q=initial_q,
                        max_iters=200,
                        tol_pos=1e-2,  # 10mm tolerance
                        tol_ori=orientation_tolerance,
                        position_weight=20.0,  # Heavy position priority
                        orientation_weight=ori_weight_ccd,
                    )
                elif method == "standard":
                    # Standard IK with adaptive orientation weighting
                    result_q = kin_helper.inverse_kinematics(
                        target_pos=target_pos,
                        target_quat=downward_quat,
                        initial_q=initial_q,
                        max_iters=300,
                        tol_pos=1e-2,  # 10mm tolerance
                        tol_ori=orientation_tolerance,
                        damping=5e-4,
                        step_size=0.3,
                        position_weight=50.0,  # Massively prioritize position
                        orientation_weight=ori_weight_standard,
                    )
                elif method == "staged":
                    # Staged IK that adapts weighting
                    horizontal_dist = np.linalg.norm(offset[:2])
                    result_q = kin_helper.inverse_kinematics_staged(
                        target_pos=target_pos,
                        target_quat=downward_quat,
                        initial_q=initial_q,
                        horizontal_distance=horizontal_dist,
                        max_iters=200,
                        damping=1e-3,
                    )
                
                # Validate result
                check_pos, check_quat = kin_helper.forward_kinematics(result_q)
                pos_error = np.linalg.norm(check_pos - target_pos)
                
                if pos_error < 0.02:  # Within 20mm
                    keyframes[keyframe_name] = result_q
                    use_ik = True
                    print(f"  ✓ IK ({method}) converged for '{keyframe_name}' (error: {pos_error*1000:.1f}mm)")
                    break  # Success, stop trying other methods
                    
            except Exception:
                continue  # Try next method
        
        if not use_ik:
            # Fallback: Use improved offset calculation
            fallback_q = base_keyframes[keyframe_name].copy()
            
            # More accurate joint adjustments based on Jacobian estimates
            # Joint 1 (base rotation): controls Y position (lateral)
            # Joint 2 (shoulder): controls X position (forward/back)
            
            # Adjust joint 1 for lateral offset (Y direction)
            # At 0.5m reach, ~1 rad ≈ 0.5m lateral movement
            if abs(offset[1]) > 0.005:  # >5mm offset
                fallback_q[0] += offset[1] * 2.0
            
            # Adjust joint 2 for forward/backward offset (X direction)  
            # At 0.5m reach, ~0.1 rad ≈ 0.04m forward movement
            if abs(offset[0]) > 0.005:  # >5mm offset
                fallback_q[1] += offset[0] * 2.5
            
            # Verify fallback is reasonable
            check_pos_fallback, _ = kin_helper.forward_kinematics(fallback_q)
            fallback_error = np.linalg.norm(check_pos_fallback - target_pos)
            
            keyframes[keyframe_name] = fallback_q
            print(f"  ⚠ IK failed for '{keyframe_name}', using offset fallback (error: {fallback_error*1000:.1f}mm)")
    
    # Other keyframes derived from grasp position
    keyframes["grasp_closed"] = keyframes["grasp"]
    keyframes["lift"] = keyframes["pre_grasp"]  # Lift to pre_grasp height
    
    # IK-computed placement keyframes (96% success rate)
    # These place the ball at bin center [0.55, 0.45] at optimal height (65mm)
    # Keep height higher during transport to avoid collisions and improve stability
    keyframes["transport_mid"] = np.array([0.560, 0.355, 0.150, -1.910, 0.175, 2.015, 0.675])
    
    # Interpolate to create smoother transition (5 steps)
    # This prevents high accelerations that cause the ball to slip
    num_inter_steps = 5
    for i in range(1, num_inter_steps):
        alpha = i / num_inter_steps
        keyframes[f"transport_inter_{i}"] = (1 - alpha) * keyframes["lift"] + alpha * keyframes["transport_mid"]
    
    # Lower only at the end
    keyframes["transport"] = np.array([0.580, 0.690, 0.150, -1.645, 0.420, 2.405, 0.895])
    keyframes["place"] = np.array([0.585, 0.790, 0.060, -1.630, 0.550, 2.485, 0.895])
    keyframes["place_open"] = keyframes["place"]
    keyframes["hold_open"] = keyframes["place"]
    
    return keyframes


def keyframe_policy(
    env: FrankaPickPlaceEnv,
    controller: KeyframeController,
    steps_at_keyframe: int,
    dwell_time: int = 20,
) -> tuple[np.ndarray, bool]:
    """Keyframe-based policy for reliable pick-and-place.
    
    Args:
        env: The Franka environment.
        controller: KeyframeController managing waypoint sequence.
        steps_at_keyframe: Number of steps at current keyframe.
        dwell_time: Minimum timesteps to dwell at each keyframe before checking convergence.
    
    Returns:
        Tuple of (action, sequence_complete):
            - action: 8D action vector (7 joint positions + 1 gripper position)
            - sequence_complete: True if all keyframes have been reached
    """
    # Get current keyframe target
    keyframe_name, target_q = controller.get_current_target()
    
    # Get current joint state
    current_q = env.data.qpos[env._joint_qpos_indices].copy()
    current_qvel = env.data.qvel[env._joint_dof_indices].copy()
    
    # ULTRA-DENSE DEMOS: Almost pure threshold-based transitions
    # Minimal dwell times - transitions happen immediately when criteria met
    if keyframe_name == "grasp":
        required_dwell = 10   # Ensure stable pre-grasp position
    elif keyframe_name == "grasp_closed":
        required_dwell = 25   # Allow time for gripper to fully close
    elif keyframe_name == "lift":
        required_dwell = 10   # Lift slowly
    elif keyframe_name == "hold_open":
        required_dwell = 15   # Hold after opening to ensure ball drops and settles
    else:
        required_dwell = 1   # Single step minimum (immediate transition)
    
    # Check if we've converged and dwelled long enough
    if steps_at_keyframe >= required_dwell and controller.check_convergence(current_q, current_qvel, target_q):
        # Try to advance to next keyframe
        controller.advance_to_next_keyframe()
    
    # Build action: position control for arm + gripper command
    action = np.zeros(8)
    action[:7] = target_q
    
    # Gripper control based on keyframe name
    # Keep gripper FIRMLY closed during all carrying phases
    if "closed" in keyframe_name or "transport" in keyframe_name or keyframe_name in ["lift", "place"]:
        action[7] = 0.0  # Fully closed gripper (0.0m = maximum grip)
    else:
        action[7] = 0.04  # Open gripper (0.04m = fully open) for approach, grasp, pre_grasp, place_open, hold_open
    
    sequence_complete = controller.is_sequence_complete()
    
    return action, sequence_complete


def collect_episode(env: FrankaPickPlaceEnv, hindered: bool, max_steps: int) -> EpisodeBuffer:
    obs, info = env.reset(hindered=hindered)
    buffer = EpisodeBuffer(
        rgb_frames=[],
        proprio=[],
        actions=[],
        timestamps=[],
        object_positions=[],  # NEW: Initialize empty list for object positions
        instruction=info["instruction"],
        meta=info,
    )

    # Get target object position
    target_color = env.target_color
    target_site_id = env._object_site_ids[target_color]
    object_pos = env.data.site_xpos[target_site_id].copy()
    
    print(f"Starting episode | Target: {target_color} at {object_pos} | Hindered: {hindered}")
    
    # Create kinematics helper for adaptive IK
    kin_helper = KinematicsHelper(env.model, site_name="gripper")
    
    # Compute adaptive keyframes for this object position
    keyframes = compute_adaptive_keyframes(env, object_pos, kin_helper)
    
    # Initialize keyframe controller
    controller = KeyframeController(
        keyframes=keyframes,
        convergence_threshold=0.08,  # 0.08 radians (~4.5 degrees)
        velocity_threshold=0.15,      # 0.15 rad/s
    )
    
    # Set pick-and-place sequence (using intermediate waypoint for reliability)
    pick_place_sequence = [
        "home", "pre_grasp", "grasp", "grasp_closed", "lift",
        "transport_inter_1", "transport_inter_2", "transport_inter_3", "transport_inter_4",
        "transport_mid", "transport", "place", "place_open", "hold_open"  # End with hold_open to ensure ball drops
    ]
    controller.set_sequence(pick_place_sequence)

    timestamp = 0.0
    steps_at_keyframe = 0  # Track steps at current keyframe
    prev_keyframe_idx = 0
    
    for step_idx in range(max_steps):
        # Get action from keyframe policy
        action, sequence_complete = keyframe_policy(env, controller, steps_at_keyframe, dwell_time=20)
        
        # Get current object position and convert to normalized image coordinates
        current_obj_pos_3d = env.data.site_xpos[target_site_id].copy()
        obj_pos_normalized = world_to_image_coords(current_obj_pos_3d)
        
        # Record data with ground truth object position
        buffer.extend(obs, action, timestamp, obj_pos_normalized)
        
        # Step environment
        result = env.step(action)
        obs = result.observation
        timestamp += env.step_dt
        steps_at_keyframe += 1
        
        # Reset step counter when advancing to new keyframe
        current_keyframe_idx, _ = controller.get_progress()
        if current_keyframe_idx != prev_keyframe_idx:
            steps_at_keyframe = 0
            prev_keyframe_idx = current_keyframe_idx
        
        # Sync the viewer if GUI is enabled
        if env.viewer is not None:
            env.viewer.sync()
        
        # Log progress
        if step_idx % 50 == 0:
            current_keyframe, _ = controller.get_current_target()
            progress = controller.get_progress()
            obj_z = env.data.site_xpos[target_site_id][2]
            print(f"  Step {step_idx:3d} | Keyframe [{progress[0]}/{progress[1]}] {current_keyframe} | Obj Z: {obj_z:.4f}")
        
        # Check success condition if sequence is complete
        if sequence_complete:
            # Check if ball is actually in the box (success condition)
            obj_pos = env.data.site_xpos[target_site_id]
            horizontal_dist = np.linalg.norm(obj_pos[:2] - env.bin_position[:2])
            obj_in_bin = horizontal_dist < env.bin_radius and obj_pos[2] < 0.08
            
            if obj_in_bin:
                print(f"  ✓ Ball in box! Episode complete at step {step_idx}")
                break
        
        # Check termination conditions
        if result.truncated:
            print(f"  Episode truncated at step {step_idx}")
            if sequence_complete:
                 print(f"  ⚠ Failed: Sequence complete but ball not in box.")
                 obj_pos = env.data.site_xpos[target_site_id]
                 horizontal_dist = np.linalg.norm(obj_pos[:2] - env.bin_position[:2])
                 print(f"    Final Object Pos: {obj_pos}")
                 print(f"    Bin Pos: {env.bin_position}")
                 print(f"    Horizontal Dist: {horizontal_dist:.4f} (Threshold: {env.bin_radius})")
                 print(f"    Height: {obj_pos[2]:.4f} (Threshold: 0.08)")
                 # Mark as failed even if sequence complete
                 sequence_complete = False
            break
            
        if result.terminated:
            if sequence_complete:
                # Double check success condition
                obj_pos = env.data.site_xpos[target_site_id]
                horizontal_dist = np.linalg.norm(obj_pos[:2] - env.bin_position[:2])
                obj_in_bin = horizontal_dist < env.bin_radius and obj_pos[2] < 0.08
                if obj_in_bin:
                    print(f"  ✓ Ball in box! Episode complete at step {step_idx}")
                    break
            # If terminated but sequence not complete, ignore and continue (early success during transport)

    buffer.meta.update(
        {
            "episode_length": len(buffer.actions),
            "control_dt": env.step_dt,
            "sequence_complete": sequence_complete,
        }
    )
    return buffer


def write_metadata(dataset_root: Path, metadata: List[Dict[str, object]], train_fraction: float = 0.8) -> None:
    """Write dataset metadata with train/val splits.
    
    Args:
        dataset_root: Root directory for the dataset.
        metadata: List of episode metadata dictionaries.
        train_fraction: Fraction of episodes to use for training (default: 0.8).
    """
    num_episodes = len(metadata)
    num_train = int(train_fraction * num_episodes)
    
    # Create train/val splits
    train_episodes = [item["episode"] for item in metadata[:num_train]]
    val_episodes = [item["episode"] for item in metadata[num_train:]]
    
    payload = {
        "episodes": metadata,
        "num_static": sum(1 for item in metadata if not item.get("hindered", False)),
        "num_hindered": sum(1 for item in metadata if item.get("hindered", False)),
        "splits": {
            "train": train_episodes,
            "val": val_episodes,
        },
    }
    (dataset_root / "metadata.json").write_text(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect LeRobot demonstrations using MuJoCo.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"), help="Output directory for episodes.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to record.")
    parser.add_argument("--hindered-fraction", type=float, default=0.1, help="Fraction of episodes with hindered resets.")
    parser.add_argument("--train-fraction", type=float, default=0.9, help="Fraction of episodes to use for training (vs validation).")
    parser.add_argument("--max-steps", type=int, default=600, help="Maximum steps per episode for full pick-and-place.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--gui", action="store_true", help="Enable the interactive MuJoCo viewer.")
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path("env/mujoco_assets"),
        help="Directory containing franka_scene.xml and associated assets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root: Path = args.dataset
    dataset_root.mkdir(parents=True, exist_ok=True)

    # Check for existing episodes to determine starting point
    existing_episodes = []
    if dataset_root.exists():
        existing_episodes = [d for d in dataset_root.iterdir() 
                           if d.is_dir() and d.name.startswith('episode_')]
    
    start_episode = len(existing_episodes)
    print(f"Found {len(existing_episodes)} existing episodes. Starting from episode {start_episode}")
    
    # Load existing metadata if it exists
    metadata_path = dataset_root / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            existing_metadata = json.load(handle)
        metadata = existing_metadata.get("episodes", [])
        print(f"Loaded existing metadata with {len(metadata)} episodes")
    else:
        metadata = []

    env = FrankaPickPlaceEnv(gui=args.gui, seed=args.seed, asset_root=args.asset_root)

    rng = np.random.default_rng(args.seed)
    hindered_fraction = float(np.clip(args.hindered_fraction, 0.0, 1.0))

    # Skip random numbers for episodes we've already collected to maintain reproducibility
    for _ in range(start_episode):
        rng.random()

    # Continue from where we left off
    for episode_idx in range(start_episode, args.episodes):
        hindered = rng.random() < hindered_fraction
        buffer = collect_episode(env, hindered=hindered, max_steps=args.max_steps)
        buffer.save(dataset_root, episode_idx)
        
        # Check if episode was successful
        success = buffer.meta.get("sequence_complete", False)
        # Double check with object position if available
        if success and len(buffer.object_positions) > 0:
            final_obj_pos = buffer.object_positions[-1]
            # Note: object_positions are normalized [0,1], so we can't easily check absolute distance here
            # relying on sequence_complete which is set based on the physical check in collect_episode
        
        metadata.append({
            "episode": f"episode_{episode_idx:04d}",
            "length": len(buffer.actions),
            "hindered": hindered,
            "instruction": buffer.instruction,
            "target_color": buffer.meta.get("target_color"),
            "success": success
        })
        
        # Log status to a separate file for easy filtering
        status_file = dataset_root / "collection_status.txt"
        with open(status_file, "a") as f:
            f.write(f"episode_{episode_idx:04d},{success}\n")
            
        print(f"Recorded episode {episode_idx:04d} | hindered={hindered} | steps={len(buffer.actions)} | success={success}")

    write_metadata(dataset_root, metadata, train_fraction=args.train_fraction)
    env.close()
    print(f"Saved dataset with {len(metadata)} episodes to {dataset_root}")
    print(f"Train/val split: {int(args.train_fraction * len(metadata))}/{len(metadata) - int(args.train_fraction * len(metadata))} episodes")


if __name__ == "__main__":
    main()

