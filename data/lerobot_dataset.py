"""Dataset utilities for loading LeRobot-format demonstrations."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

# Import object detection utility
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.object_detection import detect_object_position_from_rgb


@dataclass
class EpisodeIndex:
    """Metadata describing a single LeRobot episode."""

    path: Path
    length: int
    instruction: str
    split: str = "train"


class LeRobotDataset(Dataset):
    """PyTorch dataset for LeRobot-formatted demonstrations.

    The dataset expects the following structure within ``root``::

        root/
          episode_0001/
            rgb_static/
              000000.png
              ...
            proprio.npy
            actions.npy
            instruction.txt
            meta.json (optional)

    A top-level ``metadata.json`` file can optionally define dataset splits by
    listing episode directory names under ``{"splits": {"train": [...], ...}}``.
    """

    def __init__(
        self,
        root: Path | str,
        split: str = "train",
        modalities: Optional[Sequence[str]] = None,
        image_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        sequence_length: int = 1,
        normalize_proprio: bool = True,
        normalize_actions: bool = True,
        action_stats: Optional[Dict[str, List[float]]] = None,
        history_length: int = 5,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        if modalities is None:
            modalities = ("rgb_static", "proprio", "action", "instruction")
        self.modalities = tuple(modalities)
        self.image_transform = image_transform
        self.sequence_length = sequence_length
        self.normalize_proprio = normalize_proprio
        self.normalize_actions = normalize_actions
        self.action_stats = action_stats
        self.history_length = history_length

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        self.episodes = self._discover_episodes(split)
        if not self.episodes:
            raise RuntimeError(f"No episodes found for split '{split}' in {self.root}")

        self.index: List[Tuple[int, int]] = []
        for episode_idx, episode in enumerate(self.episodes):
            for t in range(episode.length):
                if t + self.sequence_length <= episode.length:
                    self.index.append((episode_idx, t))

    # ------------------------------------------------------------------
    # PyTorch Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor | str]:
        episode_idx, start_t = self.index[item]
        episode = self.episodes[episode_idx]
        data = self._load_episode(episode)

        end_t = start_t + self.sequence_length

        sample: Dict[str, torch.Tensor | str] = {"episode_id": episode.path.name, "timestep": start_t}
        if "rgb_static" in self.modalities:
            frames = [self._load_image(path) for path in data["rgb_static"][start_t:end_t]]
            if self.sequence_length == 1:
                sample["rgb_static"] = frames[0]
            else:
                sample["rgb_static"] = torch.stack(frames, dim=0)
            
            # Extract object position: use ground truth if available, fallback to RGB detection
            if data.get("object_positions") is not None:
                # NEW: Use ground truth object position from MuJoCo (accurate, no noise)
                obj_pos = data["object_positions"][start_t]  # Already normalized to [0,1]
                sample["object_position"] = torch.from_numpy(obj_pos).float()
            else:
                # FALLBACK: Use color-based detection from RGB (for old datasets)
                if self.sequence_length == 1:
                    rgb_for_detection = frames[0]
                else:
                    rgb_for_detection = frames[0]  # Use first frame
                
                # Convert from (C, H, W) to (H, W, C) for detection
                rgb_np = rgb_for_detection.permute(1, 2, 0).numpy()
                # Extract target color from instruction (e.g. "Pick up the red sphere...")
                instruction = episode.instruction.lower()
                if "red" in instruction:
                    target_color = "red"
                elif "green" in instruction:
                    target_color = "green"
                elif "blue" in instruction:
                    target_color = "blue"
                else:
                    target_color = "red"  # default
                
                obj_x, obj_y = detect_object_position_from_rgb(rgb_np, target_color=target_color)
                sample["object_position"] = torch.tensor([obj_x, obj_y], dtype=torch.float32)

        if "proprio" in self.modalities:
            proprio = torch.from_numpy(data["proprio"][start_t:end_t]).float()
            if self.normalize_proprio:
                # Normalize using fixed joint limits (Franka Panda: ±2.8973 rad)
                # This ensures consistent normalization across all episodes and timesteps
                joint_min = -2.8973
                joint_max = 2.8973
                proprio = 2.0 * (proprio - joint_min) / (joint_max - joint_min) - 1.0
            
            # ADD TIMESTEP as 8th dimension for temporal awareness
            # Normalize timestep to [0, 1] using FIXED max_steps to match evaluation
            # CRITICAL: Must use same normalization as evaluate_bc_mujoco.py
            # Updated for new demos: max episode length 300
            MAX_EPISODE_STEPS = 300  # Fixed constant matching environment max_steps
            timesteps = torch.arange(start_t, end_t, dtype=torch.float32) / MAX_EPISODE_STEPS
            timesteps = timesteps.unsqueeze(-1)  # Shape: (seq_len, 1)
            
            # Concatenate: (seq_len, 7) + (seq_len, 1) → (seq_len, 8)
            proprio = torch.cat([proprio, timesteps], dim=-1)
            
            sample["proprio"] = proprio if self.sequence_length > 1 else proprio.squeeze(0)

        if "action" in self.modalities:
            action = torch.from_numpy(data["actions"][start_t:end_t]).float()
            
            # Enable interpolation to generate smooth trajectories from keyframe demos
            # This converts keyframe holds into smooth motions between waypoints
            if self.sequence_length == 1 and start_t > 0 and start_t < len(data["actions"]) - 1:
                # Get previous and next actions
                prev_action = torch.from_numpy(data["actions"][start_t - 1]).float()
                next_action = torch.from_numpy(data["actions"][min(start_t + 1, len(data["actions"]) - 1)]).float()
                
                # Check if current action is same as previous (keyframe hold)
                is_holding = torch.allclose(action[0, :7], prev_action[:7], atol=1e-4)
                
                if is_holding:
                    # Find the next different action (end of keyframe hold)
                    keyframe_end = start_t + 1
                    while keyframe_end < len(data["actions"]):
                        if not torch.allclose(
                            torch.from_numpy(data["actions"][keyframe_end]).float()[:7],
                            action[0, :7],
                            atol=1e-4
                        ):
                            break
                        keyframe_end += 1
                    
                    if keyframe_end < len(data["actions"]):
                        # Interpolate between current keyframe and next keyframe
                        next_keyframe = torch.from_numpy(data["actions"][keyframe_end]).float()
                        keyframe_duration = keyframe_end - start_t
                        
                        # Linear interpolation for joints (dims 0-6)
                        alpha = min(1.0 / keyframe_duration, 1.0)  # Interpolation factor
                        action[0, :7] = action[0, :7] * (1 - alpha) + next_keyframe[:7] * alpha
                        # Keep gripper as-is (binary control)
            
            if self.normalize_actions and self.action_stats is not None:
                # Normalize actions using dataset statistics
                action_mean = torch.tensor(self.action_stats["mean"], dtype=torch.float32)
                action_std = torch.tensor(self.action_stats["std"], dtype=torch.float32)
                # Normalize ALL dimensions including gripper for consistent scale
                # This prevents scale mismatch between normalized joints (range ~[-3,+3]) 
                # and raw gripper (range [0.0,0.04]) which causes gradient instability
                action = (action - action_mean) / (action_std + 1e-8)
            sample["action"] = action if self.sequence_length > 1 else action.squeeze(0)
            
            # Add action history for temporal context (closed-loop control)
            if self.history_length > 0:
                # Get history window [start_t - history_length, start_t)
                history_start = max(0, start_t - self.history_length)
                action_history = torch.from_numpy(data["actions"][history_start:start_t]).float()
                
                # Pad with zeros if at episode start (insufficient history)
                if len(action_history) < self.history_length:
                    padding_size = self.history_length - len(action_history)
                    padding = torch.zeros((padding_size, data["actions"].shape[1]), dtype=torch.float32)
                    action_history = torch.cat([padding, action_history], dim=0)
                
                # Normalize history with same stats as current action
                if self.normalize_actions and self.action_stats is not None:
                    action_history = (action_history - action_mean) / (action_std + 1e-8)
                
                sample["action_history"] = action_history

        if "instruction" in self.modalities:
            sample["instruction"] = episode.instruction

        return sample

    # ------------------------------------------------------------------
    # Collation helpers
    # ------------------------------------------------------------------
    def collate_fn(self, batch: Sequence[Dict[str, torch.Tensor | str]]) -> Dict[str, torch.Tensor]:
        batch_tensors: Dict[str, List[torch.Tensor]] = {}
        instructions: List[str] = []
        meta: Dict[str, List[str | int]] = {"episode_id": [], "timestep": []}

        for item in batch:
            meta["episode_id"].append(item["episode_id"])
            meta["timestep"].append(int(item["timestep"]))

            if "rgb_static" in item:
                batch_tensors.setdefault("rgb_static", []).append(item["rgb_static"])  # type: ignore[arg-type]
            if "proprio" in item:
                batch_tensors.setdefault("proprio", []).append(item["proprio"])  # type: ignore[arg-type]
            if "action" in item:
                batch_tensors.setdefault("action", []).append(item["action"])  # type: ignore[arg-type]
            if "action_history" in item:
                batch_tensors.setdefault("action_history", []).append(item["action_history"])  # type: ignore[arg-type]
            if "object_position" in item:
                batch_tensors.setdefault("object_position", []).append(item["object_position"])  # type: ignore[arg-type]
            if "instruction" in item:
                instructions.append(item["instruction"])  # type: ignore[arg-type]

        collated: Dict[str, torch.Tensor | List[str] | List[str | int]] = {"meta": meta}
        for key, tensors in batch_tensors.items():
            collated[key] = torch.stack(tensors, dim=0)
        if instructions:
            collated["instruction"] = instructions
        return collated  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _discover_episodes(self, split: str) -> List[EpisodeIndex]:
        metadata_path = self.root / "metadata.json"
        split_map: Dict[str, List[str]] = {}
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            split_map = metadata.get("splits", {})

        candidates: Iterable[Path]
        if split_map and split in split_map:
            candidates = [self.root / episode for episode in split_map[split]]
        else:
            candidates = sorted(path for path in self.root.iterdir() if path.is_dir())

        episodes: List[EpisodeIndex] = []
        for episode_path in candidates:
            actions_path = episode_path / "actions.npy"
            instruction_path = episode_path / "instruction.txt"
            if not actions_path.exists() or not instruction_path.exists():
                continue
            actions = np.load(actions_path)
            length = int(actions.shape[0])
            with instruction_path.open("r", encoding="utf-8") as handle:
                instruction = handle.read().strip()
            episodes.append(EpisodeIndex(path=episode_path, length=length, instruction=instruction, split=split))
        return episodes

    def _load_episode(self, episode: EpisodeIndex) -> Dict[str, np.ndarray | List[Path]]:
        cache_key = episode.path
        if not hasattr(self, "_episode_cache"):
            self._episode_cache = {}
        if cache_key in self._episode_cache:
            return self._episode_cache[cache_key]

        # Images and proprio are in the "obs" subdirectory
        rgb_dir = episode.path / "obs" / "rgb_static"
        image_paths = []
        if rgb_dir.exists():
            image_paths = sorted(rgb_dir.glob("*.png"))
            if not image_paths:
                image_paths = sorted(rgb_dir.glob("*.jpg"))

        # Try obs/proprio.npy first (new format), fall back to proprio.npy (old format)
        proprio_path = episode.path / "obs" / "proprio.npy"
        if not proprio_path.exists():
            proprio_path = episode.path / "proprio.npy"
        
        actions_path = episode.path / "actions.npy"
        proprio = np.load(proprio_path) if proprio_path.exists() else np.zeros((episode.length, 7), dtype=np.float32)
        actions = np.load(actions_path)
        
        # NEW: Load ground truth object positions if available
        object_positions_path = episode.path / "obs" / "object_positions.npy"
        object_positions = None
        if object_positions_path.exists():
            object_positions = np.load(object_positions_path)

        data = {
            "rgb_static": image_paths, 
            "proprio": proprio, 
            "actions": actions,
            "object_positions": object_positions,  # NEW: Include object positions
        }
        self._episode_cache[cache_key] = data
        return data

    def _load_image(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        if self.image_transform is not None:
            return self.image_transform(image)
        array = torch.from_numpy(np.array(image)).float() / 255.0
        return array.permute(2, 0, 1)


def _normalize_range(tensor: torch.Tensor, low: float, high: float) -> torch.Tensor:
    minimum = tensor.min()
    maximum = tensor.max()
    if maximum == minimum:
        return torch.zeros_like(tensor)
    scaled = (tensor - minimum) / (maximum - minimum)
    return scaled * (high - low) + low


__all__ = ["LeRobotDataset", "EpisodeIndex"]
