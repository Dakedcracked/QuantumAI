"""Base classifier class for medical image classification."""

import os
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, List, Any
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import (
    ResNet50, VGG16, InceptionV3, DenseNet121, EfficientNetB0
)


class BaseClassifier(ABC):
    """Abstract base class for medical image classifiers."""
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 2,
        base_model_name: str = "ResNet50",
        learning_rate: float = 0.001,
    ):
        """
        Initialize the base classifier.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of classification classes
            base_model_name: Name of the base CNN architecture
            learning_rate: Learning rate for optimizer
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.base_model_name = base_model_name
        self.learning_rate = learning_rate
        self.model: Optional[Model] = None
        self.history: Optional[Dict] = None
        
    def build_model(self, freeze_base: bool = True) -> Model:
        """
        Build the classification model with transfer learning.
        
        Args:
            freeze_base: Whether to freeze base model weights
            
        Returns:
            Compiled Keras model
        """
        # Select base model
        base_models = {
            "ResNet50": ResNet50,
            "VGG16": VGG16,
            "InceptionV3": InceptionV3,
            "DenseNet121": DenseNet121,
            "EfficientNetB0": EfficientNetB0,
        }
        
        if self.base_model_name not in base_models:
            raise ValueError(f"Unsupported base model: {self.base_model_name}")
        
        # Load base model without top layer
        base_model = base_models[self.base_model_name](
            weights="imagenet",
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model if specified
        base_model.trainable = not freeze_base
        
        # Build custom top layers
        inputs = keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer
        if self.num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
            loss = 'categorical_crossentropy'
        
        # Create model
        model = Model(inputs, outputs)
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=loss,
            metrics=['accuracy', keras.metrics.AUC(name='auc'), 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        self.model = model
        return model
    
    def train(
        self,
        train_data,
        validation_data,
        epochs: int = 50,
        batch_size: int = 32,
        callbacks: Optional[List] = None,
        class_weights: Optional[Dict[int, float]] = None # <--- NEW PARAMETER
    ) -> Dict:
        """
        Train the classification model.
        
        Args:
            train_data: Training dataset
            validation_data: Validation dataset
            epochs: Number of training epochs
            batch_size: Batch size for training
            callbacks: List of Keras callbacks
            class_weights: Dictionary mapping class indices to weights for loss calculation
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.build_model()
        
        if callbacks is None:
            callbacks = self._get_default_callbacks()
        
        history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weights, # <--- PASSING WEIGHTS HERE
            verbose=1
        )
        
        self.history = history.history
        return self.history
    
    # [Rest of the methods: predict, evaluate, save_model, load_model, _get_default_callbacks, get_model_info]
    def predict(self, images: np.ndarray) -> np.ndarray:
        """
        Make predictions on input images.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        return self.model.predict(images)
    
    def evaluate(self, test_data) -> Dict[str, float]:
        """
        Evaluate model on test data.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        results = self.model.evaluate(test_data, verbose=0)
        metrics = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            metrics[metric_name] = results[i]
        
        return metrics
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def _get_default_callbacks(self) -> List:
        """Get default training callbacks."""
        return [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
        ]
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the model."""
        pass