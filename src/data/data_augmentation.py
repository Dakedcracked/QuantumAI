"""Data augmentation utilities."""

import numpy as np
import cv2
from typing import Tuple, Optional
import tensorflow as tf


class DataAugmentor:
    """Advanced data augmentation for medical images."""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize data augmentor.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def random_rotation(
        self,
        image: np.ndarray,
        max_angle: float = 20.0
    ) -> np.ndarray:
        """
        Apply random rotation to image.
        
        Args:
            image: Input image
            max_angle: Maximum rotation angle in degrees
            
        Returns:
            Rotated image
        """
        angle = np.random.uniform(-max_angle, max_angle)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
        
        return rotated
    
    def random_flip(
        self,
        image: np.ndarray,
        horizontal: bool = True,
        vertical: bool = False
    ) -> np.ndarray:
        """
        Apply random flipping to image.
        
        Args:
            image: Input image
            horizontal: Enable horizontal flip
            vertical: Enable vertical flip
            
        Returns:
            Flipped image
        """
        if horizontal and np.random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        if vertical and np.random.random() > 0.5:
            image = cv2.flip(image, 0)
        
        return image
    
    def random_zoom(
        self,
        image: np.ndarray,
        zoom_range: Tuple[float, float] = (0.8, 1.2)
    ) -> np.ndarray:
        """
        Apply random zoom to image.
        
        Args:
            image: Input image
            zoom_range: Range of zoom factors (min, max)
            
        Returns:
            Zoomed image
        """
        zoom_factor = np.random.uniform(zoom_range[0], zoom_range[1])
        h, w = image.shape[:2]
        
        # Calculate new dimensions
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Crop or pad to original size
        if zoom_factor > 1.0:
            # Crop center
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            result = resized[start_h:start_h + h, start_w:start_w + w]
        else:
            # Pad with reflection
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            result = cv2.copyMakeBorder(
                resized, pad_h, h - new_h - pad_h, pad_w, w - new_w - pad_w,
                cv2.BORDER_REFLECT
            )
        
        return result
    
    def random_shift(
        self,
        image: np.ndarray,
        max_shift: float = 0.2
    ) -> np.ndarray:
        """
        Apply random shift to image.
        
        Args:
            image: Input image
            max_shift: Maximum shift as fraction of image dimensions
            
        Returns:
            Shifted image
        """
        h, w = image.shape[:2]
        
        shift_h = int(np.random.uniform(-max_shift, max_shift) * h)
        shift_w = int(np.random.uniform(-max_shift, max_shift) * w)
        
        translation_matrix = np.float32([[1, 0, shift_w], [0, 1, shift_h]])
        shifted = cv2.warpAffine(image, translation_matrix, (w, h),
                                 borderMode=cv2.BORDER_REFLECT)
        
        return shifted
    
    def random_brightness(
        self,
        image: np.ndarray,
        brightness_range: Tuple[float, float] = (0.8, 1.2)
    ) -> np.ndarray:
        """
        Apply random brightness adjustment.
        
        Args:
            image: Input image
            brightness_range: Range of brightness factors (min, max)
            
        Returns:
            Brightness-adjusted image
        """
        factor = np.random.uniform(brightness_range[0], brightness_range[1])
        adjusted = np.clip(image * factor, 0, 255).astype(image.dtype)
        return adjusted
    
    def random_contrast(
        self,
        image: np.ndarray,
        contrast_range: Tuple[float, float] = (0.8, 1.2)
    ) -> np.ndarray:
        """
        Apply random contrast adjustment.
        
        Args:
            image: Input image
            contrast_range: Range of contrast factors (min, max)
            
        Returns:
            Contrast-adjusted image
        """
        factor = np.random.uniform(contrast_range[0], contrast_range[1])
        mean = np.mean(image)
        adjusted = np.clip((image - mean) * factor + mean, 0, 255).astype(image.dtype)
        return adjusted
    
    def add_gaussian_noise(
        self,
        image: np.ndarray,
        mean: float = 0.0,
        std: float = 5.0
    ) -> np.ndarray:
        """
        Add Gaussian noise to image.
        
        Args:
            image: Input image
            mean: Mean of Gaussian noise
            std: Standard deviation of Gaussian noise
            
        Returns:
            Noisy image
        """
        noise = np.random.normal(mean, std, image.shape)
        noisy = np.clip(image + noise, 0, 255).astype(image.dtype)
        return noisy
    
    def augment(
        self,
        image: np.ndarray,
        apply_rotation: bool = True,
        apply_flip: bool = True,
        apply_zoom: bool = True,
        apply_shift: bool = True,
        apply_brightness: bool = True,
        apply_contrast: bool = True,
        apply_noise: bool = False
    ) -> np.ndarray:
        """
        Apply multiple augmentations to image.
        
        Args:
            image: Input image
            apply_rotation: Apply random rotation
            apply_flip: Apply random flip
            apply_zoom: Apply random zoom
            apply_shift: Apply random shift
            apply_brightness: Apply random brightness
            apply_contrast: Apply random contrast
            apply_noise: Add Gaussian noise
            
        Returns:
            Augmented image
        """
        augmented = image.copy()
        
        if apply_rotation:
            augmented = self.random_rotation(augmented)
        
        if apply_flip:
            augmented = self.random_flip(augmented)
        
        if apply_zoom:
            augmented = self.random_zoom(augmented)
        
        if apply_shift:
            augmented = self.random_shift(augmented)
        
        if apply_brightness:
            augmented = self.random_brightness(augmented)
        
        if apply_contrast:
            augmented = self.random_contrast(augmented)
        
        if apply_noise:
            augmented = self.add_gaussian_noise(augmented)
        
        return augmented
    
    def create_augmented_dataset(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        augmentation_factor: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create augmented dataset by generating multiple versions of each image.
        
        Args:
            images: Input images
            labels: Corresponding labels
            augmentation_factor: Number of augmented versions per image
            
        Returns:
            Tuple of (augmented_images, augmented_labels)
        """
        augmented_images = []
        augmented_labels = []
        
        for img, label in zip(images, labels):
            # Add original
            augmented_images.append(img)
            augmented_labels.append(label)
            
            # Add augmented versions
            for _ in range(augmentation_factor):
                aug_img = self.augment(img)
                augmented_images.append(aug_img)
                augmented_labels.append(label)
        
        return np.array(augmented_images), np.array(augmented_labels)
