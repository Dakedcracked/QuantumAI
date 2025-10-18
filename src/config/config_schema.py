"""Pydantic schema for configuration validation."""

from typing import Tuple, List, Optional
from pydantic import BaseModel, Field

# Define the base model for strict configuration validation
class ModelSettings(BaseModel):
    """
    Schema for validating model configuration parameters.
    """
    # Core Model Parameters
    model_name: str = Field(..., description="Unique name for the trained model file.")
    model_type: str = Field(..., description="The Python class name to instantiate (e.g., EffResNetViTClassifier).")
    num_classes: int = Field(..., description="Number of output classes (e.g., 2 for binary, 4 for multi-class).")
    base_model: str = Field(..., description="The CNN feature extractor to use (e.g., ResNet50, EffResNet).")
    
    # Data Parameters
    input_shape: Tuple[int, int, int] = Field((224, 224, 3), description="Input image size (H, W, C).")
    data_dir: Optional[str] = Field(None, description="Root directory for training data.")
    test_dir: Optional[str] = Field(None, description="Root directory for test data.")
    validation_split: float = Field(0.2, description="Fraction of data to reserve for validation.")
    class_labels: List[str] = Field(..., description="List of human-readable class names.")

    # Training Hyperparameters
    learning_rate: float = Field(0.0001, description="Initial learning rate for the optimizer.")
    batch_size: int = Field(32, description="Training batch size.")
    epochs: int = Field(50, description="Maximum number of epochs to train.")
    freeze_base: bool = Field(True, description="Whether to freeze the base CNN weights.")
    use_augmentation: bool = Field(True, description="Whether to apply data augmentation during training.")
    early_stopping_patience: int = Field(10, description="Patience for early stopping callback.")
    reduce_lr_patience: int = Field(5, description="Patience for ReduceLROnPlateau callback.")

    class Config:
        # Pydantic setting to allow tuples/lists to be validated correctly
        arbitrary_types_allowed = True