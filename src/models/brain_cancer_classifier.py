"""Brain cancer/tumor classification model."""

from typing import Dict
from src.models.base_classifier import BaseClassifier


class BrainCancerClassifier(BaseClassifier):
    """
    Specialized classifier for brain tumor/cancer detection from MRI images.
    
    Supports classification of:
    - Normal vs Tumor
    - Multi-class: Glioma, Meningioma, Pituitary, No Tumor
    """
    
    def __init__(
        self,
        input_shape=(224, 224, 3),
        num_classes=2,
        base_model_name="ResNet50",
        learning_rate=0.0001,
    ):
        """
        Initialize brain cancer classifier.
        
        Args:
            input_shape: Input image dimensions (height, width, channels)
            num_classes: Number of classes (2 for binary, 4 for standard tumor types)
            base_model_name: Base CNN architecture (ResNet50, VGG16, etc.)
            learning_rate: Learning rate for optimizer
        """
        super().__init__(
            input_shape=input_shape,
            num_classes=num_classes,
            base_model_name=base_model_name,
            learning_rate=learning_rate
        )
        
        # Brain tumor specific class labels
        if num_classes == 2:
            self.class_labels = ["No Tumor", "Tumor"]
        elif num_classes == 4:
            self.class_labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
        else:
            self.class_labels = [f"Class_{i}" for i in range(num_classes)]
    
    def get_model_info(self) -> Dict[str, str]:
        """
        Get information about the brain cancer classifier.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_type": "Brain Cancer/Tumor Classifier",
            "base_architecture": self.base_model_name,
            "input_shape": str(self.input_shape),
            "num_classes": str(self.num_classes),
            "class_labels": ", ".join(self.class_labels),
            "task": "Brain Tumor Detection and Classification",
            "imaging_modality": "MRI (T1, T2, FLAIR)",
            "description": "Deep learning model for automated brain tumor detection and classification from MRI scans",
            "Developed by": "Varun Agrawal"
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
