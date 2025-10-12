"""Image preprocessing utilities."""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path
import tensorflow as tf


class ImagePreprocessor:
    """Preprocessing utilities for medical images."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        clahe: bool = False,
    ):
        """
        Initialize the image preprocessor.
        
        Args:
            target_size: Target image size (width, height)
            normalize: Whether to normalize pixel values to [0, 1]
            clahe: Whether to apply CLAHE for contrast enhancement
        """
        self.target_size = target_size
        self.normalize = normalize
        self.clahe = clahe
        
        if self.clahe:
            self.clahe_processor = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess a single image.
        
        Args:
            image: Input image array
            
        Returns:
            Preprocessed image
        """
        # Resize image
        if image.shape[:2] != self.target_size:
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Apply CLAHE if enabled
        if self.clahe:
            if len(image.shape) == 2:
                image = self.clahe_processor.apply(image)
            elif len(image.shape) == 3:
                # Apply to each channel
                channels = cv2.split(image)
                channels = [self.clahe_processor.apply(ch) for ch in channels]
                image = cv2.merge(channels)
        
        # Normalize
        if self.normalize:
            image = image.astype(np.float32) / 255.0
        
        # Ensure 3 channels
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        return image
    
    def load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess an image from file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image
        """
        # Load image
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        return self.preprocess_image(image)
    
    def preprocess_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            Batch of preprocessed images
        """
        preprocessed = [self.preprocess_image(img) for img in images]
        return np.array(preprocessed)
    
    def denormalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Denormalize image for visualization.
        
        Args:
            image: Normalized image
            
        Returns:
            Denormalized image in [0, 255] range
        """
        if self.normalize:
            image = (image * 255).astype(np.uint8)
        return image
    
    @staticmethod
    def remove_noise(image: np.ndarray, method: str = "bilateral") -> np.ndarray:
        """
        Remove noise from image.
        
        Args:
            image: Input image
            method: Denoising method ('bilateral', 'gaussian', 'median')
            
        Returns:
            Denoised image
        """
        if method == "bilateral":
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif method == "gaussian":
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif method == "median":
            return cv2.medianBlur(image, 5)
        else:
            raise ValueError(f"Unknown denoising method: {method}")
    
    @staticmethod
    def enhance_contrast(image: np.ndarray, method: str = "clahe") -> np.ndarray:
        """
        Enhance image contrast.
        
        Args:
            image: Input image
            method: Enhancement method ('clahe', 'histogram_eq')
            
        Returns:
            Contrast-enhanced image
        """
        if method == "clahe":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            if len(image.shape) == 2:
                return clahe.apply(image)
            else:
                channels = cv2.split(image)
                channels = [clahe.apply(ch) for ch in channels]
                return cv2.merge(channels)
        elif method == "histogram_eq":
            if len(image.shape) == 2:
                return cv2.equalizeHist(image)
            else:
                channels = cv2.split(image)
                channels = [cv2.equalizeHist(ch) for ch in channels]
                return cv2.merge(channels)
        else:
            raise ValueError(f"Unknown enhancement method: {method}")
