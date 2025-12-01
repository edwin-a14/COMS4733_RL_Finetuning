"""Visual augmentation transforms for robotic manipulation."""
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import random


class RobotVisionAugmentation:
    """Augmentation pipeline for robot vision data.
    
    Applies random transformations to force the model to learn robust visual features
    instead of memorizing exact pixel patterns.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.05,
        random_crop_scale: tuple = (0.9, 1.0),
        random_rotation: float = 5.0,
        random_erasing_prob: float = 0.1,
        enabled: bool = True,
    ):
        """
        Args:
            image_size: Target image size (assumes square images)
            brightness: Random brightness adjustment range
            contrast: Random contrast adjustment range
            saturation: Random saturation adjustment range
            hue: Random hue adjustment range
            random_crop_scale: Scale range for random resized crop
            random_rotation: Maximum rotation in degrees
            random_erasing_prob: Probability of random erasing
            enabled: Whether to apply augmentation (False for validation)
        """
        self.image_size = image_size
        self.enabled = enabled
        
        if enabled:
            # Color jitter
            self.color_jitter = T.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            )
            
            # Random crop parameters
            self.crop_scale = random_crop_scale
            
            # Random rotation
            self.rotation_degrees = random_rotation
            
            # Random erasing (occlusion)
            self.random_erasing_prob = random_erasing_prob
            self.random_erasing = T.RandomErasing(
                p=random_erasing_prob,
                scale=(0.02, 0.1),
                ratio=(0.3, 3.3),
            )
        
        # Always normalize (even for validation)
        # DINOv2 expects ImageNet normalization
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __call__(self, img: Image.Image) -> torch.Tensor:
        """Apply augmentation to PIL Image.
        
        Args:
            img: PIL Image (RGB)
            
        Returns:
            Augmented image tensor (C, H, W) normalized to ImageNet stats
        """
        if not self.enabled:
            # Validation: just resize and normalize
            img = TF.resize(img, (self.image_size, self.image_size))
            img = TF.to_tensor(img)
            img = self.normalize(img)
            return img
        
        # Training: apply augmentations
        
        # 1. Random resized crop (slight zoom/crop)
        # DISABLED: Spatial augmentations break the mapping between pixels and absolute joint actions!
        # if random.random() > 0.3:  # Apply 70% of the time
        #     i, j, h, w = T.RandomResizedCrop.get_params(
        #         img,
        #         scale=self.crop_scale,
        #         ratio=(0.95, 1.05)  # Keep roughly square
        #     )
        #     img = TF.resized_crop(img, i, j, h, w, (self.image_size, self.image_size))
        # else:
        #     img = TF.resize(img, (self.image_size, self.image_size))
        
        # Always resize to target size
        img = TF.resize(img, (self.image_size, self.image_size))
        
        # 2. Random rotation (small angles)
        # DISABLED: Spatial augmentations break the mapping between pixels and absolute joint actions!
        # if random.random() > 0.5:
        #     angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        #     img = TF.rotate(img, angle, fill=(128, 128, 128))
        
        # 3. Color jitter
        if random.random() > 0.3:  # Apply 70% of the time
            img = self.color_jitter(img)
        
        # Convert to tensor
        img = TF.to_tensor(img)
        
        # 4. Random erasing (occlusion)
        if random.random() < self.random_erasing_prob:
            img = self.random_erasing(img)
        
        # 5. Normalize
        img = self.normalize(img)
        
        return img


def get_train_augmentation(image_size: int = 224, strong_color_aug: bool = False) -> RobotVisionAugmentation:
    """Get augmentation transform for training.
    
    Args:
        image_size: Target size
        strong_color_aug: Use stronger color augmentation to help frozen vision encoder
                         learn color-invariant features
    """
    if strong_color_aug:
        # STRONG color augmentation for frozen vision encoder
        # Forces model to learn robust color features despite frozen DINOv2
        return RobotVisionAugmentation(
            image_size=image_size,
            brightness=0.3,      # Increased from 0.2
            contrast=0.3,        # Increased from 0.2
            saturation=0.4,      # Increased from 0.2 (more color variation)
            hue=0.15,            # Increased from 0.05 (CRITICAL for color discrimination)
            random_crop_scale=(0.85, 1.0),  # Slightly more aggressive
            random_rotation=8.0,  # Slightly more rotation
            random_erasing_prob=0.15,  # More occlusion
            enabled=True,
        )
    else:
        # Standard augmentation
        return RobotVisionAugmentation(
            image_size=image_size,
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
            random_crop_scale=(0.9, 1.0),
            random_rotation=5.0,
            random_erasing_prob=0.1,
            enabled=True,
        )


def get_val_augmentation(image_size: int = 224) -> RobotVisionAugmentation:
    """Get augmentation transform for validation (no augmentation, just normalize)."""
    return RobotVisionAugmentation(
        image_size=image_size,
        enabled=False,
    )

