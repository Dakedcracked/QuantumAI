"""Lung cancer classification model."""

from typing import Dict
from src.models.base_classifier import BaseClassifier


class LungCancerClassifier(BaseClassifier):
    """
    Specialized classifier for lung cancer detection from CT/X-ray images.
    
    Supports classification of:
    - Normal vs Cancerous
    - Multi-class: Normal, Benign, Malignant, Adenocarcinoma, Squamous Cell Carcinoma
    """
    
    def __init__(
        self,
        input_shape=(224, 224, 3),
        num_classes=2,
        base_model_name="ResNet50",
        learning_rate=0.0001,
    ):
        """
        Initialize lung cancer classifier.
        
        Args:
            input_shape: Input image dimensions (height, width, channels)
            num_classes: Number of classes (2 for binary, more for multi-class)
            base_model_name: Base CNN architecture (ResNet50, VGG16, etc.)
            learning_rate: Learning rate for optimizer
        """
        super().__init__(
            input_shape=input_shape,
            num_classes=num_classes,
            base_model_name=base_model_name,
            learning_rate=learning_rate
        )
        
        # Lung cancer specific class labels
        if num_classes == 2:
            self.class_labels = ["Normal", "Cancerous"]
        elif num_classes == 3:
            self.class_labels = ["Normal", "Benign", "Malignant"]
        elif num_classes == 4:
            self.class_labels = ["Normal", "Adenocarcinoma", "Large Cell Carcinoma", "Squamous Cell Carcinoma"]
        else:
            self.class_labels = [f"Class_{i}" for i in range(num_classes)]
    
    def get_model_info(self) -> Dict[str, str]:
        """
        Get information about the lung cancer classifier.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_type": "Lung Cancer Classifier",
            "base_architecture": self.base_model_name,
            "input_shape": str(self.input_shape),
            "num_classes": str(self.num_classes),
            "class_labels": ", ".join(self.class_labels),
            "task": "Lung Cancer Detection and Classification",
            "imaging_modality": "CT Scan / X-Ray",
            "description": "Deep learning model for automated lung cancer detection and classification from medical images",
        }
    
    def predict_with_labels(self, images):
        """
        Make predictions and return class labels.
        
        Args:
            images: Input images
            
        Returns:
            Tuple of (predictions, class_labels)
        """
        predictions = self.predict(images)
        
        if self.num_classes == 2:
            predicted_classes = (predictions > 0.5).astype(int).flatten()
        else:
            predicted_classes = predictions.argmax(axis=1)
        
        predicted_labels = [self.class_labels[idx] for idx in predicted_classes]
        
        return predictions, predicted_labels
