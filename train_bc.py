"""Behavioral cloning training entry-point for the VLA model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

from data.lerobot_dataset import LeRobotDataset
from data.augmentation import get_train_augmentation, get_val_augmentation
from models.vla_dinov2 import VLADinoV2Config, VLADinoV2Policy
from utils.config import load_config, save_config
from utils.logging import MetricTracker, get_logger, setup_logging
from utils.seed import seed_everything


def build_dataloader(
    dataset_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    split: str,
    action_stats: Dict[str, Any] | None = None,
    use_augmentation: bool = True,
) -> Tuple[LeRobotDataset, DataLoader]:
    dataset_root = dataset_cfg["path"]
    sequence_length = dataset_cfg.get("sequence_length", 1)
    normalize_proprio = dataset_cfg.get("normalize_proprio", True)

    # Use augmentation for training, not for validation
    is_train = split == dataset_cfg.get("train_split", "train")
    if use_augmentation and is_train:
        # Use strong color augmentation if vision encoder is frozen
        strong_color_aug = model_cfg.get("freeze_vision", True)
        image_transform = get_train_augmentation(image_size=224, strong_color_aug=strong_color_aug)
    else:
        image_transform = get_val_augmentation(image_size=224)

    dataset = LeRobotDataset(
        root=dataset_root,
        split=split,
        sequence_length=sequence_length,
        image_transform=image_transform,
        normalize_proprio=normalize_proprio,
        normalize_actions=True,
        action_stats=action_stats,
        history_length=model_cfg.get("history_length", 5),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=dataset_cfg["batch_size"],
        shuffle=is_train,
        num_workers=dataset_cfg.get("num_workers", 4),
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    return dataset, dataloader


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the DINOv2+BERT VLA model via behavioral cloning.")
    parser.add_argument("--config", type=str, default="configs/openvla_dinov2_bc.yaml")
    parser.add_argument("--override", type=str, nargs="*", default=None)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs declared in the YAML config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Load a single batch and execute a forward/backward pass without writing checkpoints. "
            "Useful for smoke-testing the setup."
        ),
    )
    args = parser.parse_args()

    raw_config = load_config(args.config, overrides=args.override)
    dataset_cfg: Dict[str, Any] = dict(raw_config.get("dataset", {}))
    model_cfg: Dict[str, Any] = dict(raw_config.get("model", {}))
    training_cfg: Dict[str, Any] = dict(raw_config.get("training", {}))

    if args.epochs is not None:
        training_cfg["epochs"] = args.epochs

    setup_logging()
    logger = get_logger("train_bc")

    seed_everything(training_cfg.get("seed", 0), deterministic_cudnn=training_cfg.get("deterministic", False))

    device = torch.device(training_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    logger.info("Using device %s", device)

    # Load action statistics for normalization
    action_stats_path = Path(dataset_cfg["path"]) / "action_stats.json"
    if action_stats_path.exists():
        with action_stats_path.open("r", encoding="utf-8") as f:
            action_stats = json.load(f)
        logger.info("Loaded action statistics from %s", action_stats_path)
    else:
        logger.warning("Action statistics not found at %s. Actions will not be normalized.", action_stats_path)
        action_stats = None

    _, train_loader = build_dataloader(dataset_cfg, model_cfg, dataset_cfg.get("train_split", "train"), action_stats)
    _, val_loader = build_dataloader(dataset_cfg, model_cfg, dataset_cfg.get("val_split", "val"), action_stats)

    model_config = VLADinoV2Config(**model_cfg)
    model = VLADinoV2Policy(model_config).to(device)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=training_cfg["lr"],
        weight_decay=training_cfg.get("weight_decay", 0.0),
    )
    scheduler = (
        optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_cfg["epochs"])
        if training_cfg.get("scheduler", "cosine") == "cosine"
        else None
    )
    criterion = nn.MSELoss()
    bce_criterion = nn.BCEWithLogitsLoss()

    checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "./runs"))
    if not args.dry_run:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_config(raw_config, checkpoint_dir / f"{training_cfg.get('checkpoint_name', 'bc_run')}_config.yaml")

    if args.dry_run:
        logger.info("Running dry run with a single batch from the training loader.")
        batch = next(iter(train_loader))
        model.train()
        optimizer.zero_grad()
        rgb = batch["rgb_static"].to(device)
        proprio = batch["proprio"].to(device)
        target = batch["action"].to(device)
        instructions = batch.get("instruction") or [""] * rgb.size(0)
        outputs = model(rgb_static=rgb, proprio=proprio, instruction=instructions)
        
        # Use weighted loss (same as training)
        joint_loss = criterion(outputs[:, :7], target[:, :7])
        gripper_loss = criterion(outputs[:, 7:8], target[:, 7:8])
        loss = joint_loss + 50.0 * gripper_loss  # Balanced weight for gripper learning
        
        loss.backward()
        grad_clip = training_cfg.get("grad_clip", 0.0)
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        logger.info("Dry run completed successfully (loss=%.5f, joint=%.5f, gripper=%.5f).", 
                   float(loss.item()), float(joint_loss.item()), float(gripper_loss.item()))
        return

    # Mixed precision training setup
    use_amp = training_cfg.get("mixed_precision", False) and device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    if use_amp:
        logger.info("Mixed precision training enabled")

    best_val_loss = float("inf")

    for epoch in range(1, training_cfg["epochs"] + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            bce_criterion,
            optimizer,
            device,
            training_cfg.get("log_interval", 10),
            training_cfg.get("grad_clip", 0.0),
            scaler=scaler,
            epoch=epoch,
            total_epochs=training_cfg["epochs"],
            proprio_dropout=training_cfg.get("proprio_dropout", 0.0),
        )
        logger.info("Epoch %d | train_loss=%.5f", epoch, train_loss)

        if scheduler is not None:
            scheduler.step()

        if (epoch % training_cfg.get("val_interval", 1)) == 0:
            val_loss = evaluate(model, val_loader, criterion, bce_criterion, device, epoch=epoch, total_epochs=training_cfg["epochs"])
            logger.info("Epoch %d | val_loss=%.5f", epoch, val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = checkpoint_dir / f"{training_cfg.get('checkpoint_name', 'bc_run')}_best.pt"
                logger.info("New best checkpoint at %s", checkpoint_path)
                torch.save({
                    "model_state": model.state_dict(),
                    "config": model_config.__dict__,
                    "action_stats": action_stats,
                    "uses_bce_gripper": True
                }, checkpoint_path)

        last_checkpoint_path = checkpoint_dir / f"{training_cfg.get('checkpoint_name', 'bc_run')}_last.pt"
        torch.save({
            "model_state": model.state_dict(),
            "config": model_config.__dict__,
            "action_stats": action_stats,
            "uses_bce_gripper": True
        }, last_checkpoint_path)


def train_one_epoch(
    model: VLADinoV2Policy,
    dataloader: DataLoader,
    criterion: nn.Module,
    bce_criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    log_interval: int,
    grad_clip: float,
    scaler: Optional[GradScaler] = None,
    epoch: int = 1,
    total_epochs: int = 35,
    proprio_dropout: float = 0.0,
) -> float:
    model.train()
    tracker = MetricTracker()
    use_amp = scaler is not None

    for step, batch in enumerate(dataloader, start=1):
        rgb = batch["rgb_static"].to(device)
        proprio = batch["proprio"].to(device)
        target = batch["action"].to(device)
        action_history = batch.get("action_history")
        if action_history is not None:
            action_history = action_history.to(device)
        instructions = batch.get("instruction")
        if instructions is None:
            instructions = [""] * rgb.size(0)

        # Apply Proprioception Dropout
        if proprio_dropout > 0.0:
            mask = (torch.rand(proprio.size(0), device=device) > proprio_dropout).float().unsqueeze(1)
            proprio = proprio * mask

        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast(enabled=use_amp):
            pred = model(rgb_static=rgb, proprio=proprio, instruction=instructions, action_history=action_history)
            
            # Compute weighted loss with HIGHER gripper weighting
            gripper_weight = 10.0
            
            joint_loss = criterion(pred[:, :7], target[:, :7])
            
            # BCE Loss for Gripper
            # Threshold for normalized target: (0.02 - 0.0145) / 0.0192 = 0.286
            # We use 0.286 as the threshold to determine if gripper should be open (1) or closed (0)
            gripper_target_bin = (target[:, 7:8] > 0.286).float()
            gripper_loss = bce_criterion(pred[:, 7:8], gripper_target_bin)
            
            # Multi-task learning: Add object detection auxiliary loss
            # This forces the model to learn explicit visual grounding before action prediction
            object_detection_loss = torch.tensor(0.0, device=device)
            if hasattr(model, 'object_detection_head') and model.object_detection_head is not None:
                object_pos_gt = batch.get("object_position")
                if object_pos_gt is not None:
                    object_pos_gt = object_pos_gt.to(device)
                    object_pos_pred = model.predict_object_position(
                        rgb_static=rgb, 
                        proprio=proprio, 
                        instruction=instructions, 
                        action_history=action_history
                    )
                    # MSE loss for 2D position regression
                    object_detection_loss = criterion(object_pos_pred, object_pos_gt)
                    # Weight from config (default: 10.0)
                    object_detection_weight = model.config.object_detection_weight
                else:
                    object_detection_weight = 0.0
            else:
                object_detection_weight = 0.0
            
            # DISABLED: Smoothness loss can cause training instability
            # Will re-enable after confirming base training works
            # smoothness_loss = 0.0
            # if pred.size(0) > 1:
            #     action_diffs = pred[1:, :7] - pred[:-1, :7]
            #     smoothness_loss = (action_diffs ** 2).mean()
            
            # Combined loss: action prediction + object detection
            loss = joint_loss + gripper_weight * gripper_loss + object_detection_weight * object_detection_loss
        
        # Safety: Skip batch if loss is non-finite (NaN or Inf)
        if not torch.isfinite(loss):
            get_logger("train_bc").warning(f"Skipping batch with non-finite loss at epoch {epoch}, step {step}")
            get_logger("train_bc").warning(f"  joint_loss={joint_loss.item():.4f}, gripper_loss={gripper_loss.item():.4f}, obj_det_loss={object_detection_loss.item():.4f}")
            continue
        
        # Backward pass with gradient scaling
        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        tracker.update(loss.item(), rgb.size(0))

        if step % log_interval == 0:
            get_logger("train_bc").info("Step %d | loss=%.5f", step, tracker.average)

    return tracker.average


def evaluate(
    model: VLADinoV2Policy, 
    dataloader: DataLoader, 
    criterion: nn.Module, 
    bce_criterion: nn.Module,
    device: torch.device,
    epoch: int = 1,
    total_epochs: int = 35,
) -> float:
    model.eval()
    tracker = MetricTracker()
    with torch.no_grad():
        for batch in dataloader:
            rgb = batch["rgb_static"].to(device)
            proprio = batch["proprio"].to(device)
            target = batch["action"].to(device)
            action_history = batch.get("action_history")
            if action_history is not None:
                action_history = action_history.to(device)
            instructions = batch.get("instruction")
            if instructions is None:
                instructions = [""] * rgb.size(0)

            pred = model(rgb_static=rgb, proprio=proprio, instruction=instructions, action_history=action_history)
            
            # Use same high gripper weight as training
            gripper_weight = 10.0
            
            joint_loss = criterion(pred[:, :7], target[:, :7])
            
            # BCE Loss for Gripper
            gripper_target_bin = (target[:, 7:8] > 0.286).float()
            gripper_loss = bce_criterion(pred[:, 7:8], gripper_target_bin)
            
            # Multi-task learning: Add object detection auxiliary loss (same as training)
            object_detection_loss = torch.tensor(0.0, device=device)
            if hasattr(model, 'object_detection_head') and model.object_detection_head is not None:
                object_pos_gt = batch.get("object_position")
                if object_pos_gt is not None:
                    object_pos_gt = object_pos_gt.to(device)
                    object_pos_pred = model.predict_object_position(
                        rgb_static=rgb, 
                        proprio=proprio, 
                        instruction=instructions, 
                        action_history=action_history
                    )
                    object_detection_loss = criterion(object_pos_pred, object_pos_gt)
                    object_detection_weight = model.config.object_detection_weight
                else:
                    object_detection_weight = 0.0
            else:
                object_detection_weight = 0.0
            
            # Smoothness disabled (same as training)
            loss = joint_loss + gripper_weight * gripper_loss + object_detection_weight * object_detection_loss
            
            tracker.update(loss.item(), rgb.size(0))
    return tracker.average


if __name__ == "__main__":
    main()
