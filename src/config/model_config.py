"""Model configuration management."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from .config_schema import ModelSettings # <-- Import Pydantic Schema

class ModelConfig:
    """Configuration manager for model training and inference."""
    
    # Default configuration dictionaries (simplified, Pydantic will validate structure)
    DEFAULT_LUNG_CONFIG = {
        "model_name": "lung_cancer_classifier_resnet",
        "model_type": "LungCancerClassifier",
        "num_classes": 2,
        "base_model": "ResNet50",
        "class_labels": ["Normal", "Cancerous"],
        "data_dir": "data/lung_cancer/train",
        "test_dir": "data/lung_cancer/test",
        # Other optional fields will use Pydantic defaults
    }
    
    DEFAULT_BRAIN_CONFIG = {
        "model_name": "brain_cancer_classifier_resnet",
        "model_type": "BrainCancerClassifier",
        "num_classes": 4,
        "base_model": "ResNet50",
        "class_labels": ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
        "data_dir": "data/brain_cancer/train",
        "test_dir": "data/brain_cancer/test",
    }
    
    # Hybrid settings for the Brain Cancer task (Example: EffResNet-ViT)
    DEFAULT_BRAIN_HYBRID_CONFIG = {
        "model_name": "brain_effresnet_vit_classifier",
        "model_type": "EffResNetViTClassifier",
        "num_classes": 4, 
        "base_model": "EffResNet", 
        "learning_rate": 0.00005, 
        "batch_size": 16, 
        "epochs": 75,
        "early_stopping_patience": 12,
        "reduce_lr_patience": 6,
        "class_labels": ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
        "data_dir": "data/brain_cancer/train",
        "test_dir": "data/brain_cancer/test",
    }
    
    # Hybrid settings for the Lung Cancer task (Example: EffResNet-ViT)
    DEFAULT_LUNG_HYBRID_CONFIG = {
        "model_name": "lung_effresnet_vit_classifier",
        "model_type": "EffResNetViTClassifier",
        "num_classes": 2,
        "base_model": "EffResNet", 
        "learning_rate": 0.00005,
        "batch_size": 16, 
        "epochs": 75,
        "early_stopping_patience": 12,
        "reduce_lr_patience": 6,
        "class_labels": ["Normal", "Cancerous"],
        "data_dir": "data/lung_cancer/train",
        "test_dir": "data/lung_cancer/test",
    }

    # Initialize with a validated Pydantic object
    def __init__(self, config_data: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration using Pydantic validation.
        """
        if isinstance(config_data, ModelSettings):
            self.settings = config_data
        elif config_data is not None:
            self.settings = ModelSettings(**config_data)
        else:
            # Create a minimal default settings object if none is provided
            self.settings = ModelSettings(
                model_name="default_model", 
                num_classes=2, 
                base_model="ResNet50", 
                class_labels=["0", "1"]
            )
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ModelConfig":
        """
        Load configuration from YAML file and validate it.
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Validate configuration using Pydantic schema
        validated_settings = ModelSettings(**config_dict)
        return cls(validated_settings)
    
    # --- DEFAULT CONSTRUCTOR METHODS ---

    @classmethod
    def lung_cancer_default(cls) -> "ModelConfig":
        return cls(cls.DEFAULT_LUNG_CONFIG)
    
    @classmethod
    def brain_cancer_default(cls) -> "ModelConfig":
        return cls(cls.DEFAULT_BRAIN_CONFIG)

    @classmethod
    def lung_hybrid_default(cls) -> "ModelConfig":
        return cls(cls.DEFAULT_LUNG_HYBRID_CONFIG)
    
    @classmethod
    def brain_hybrid_default(cls) -> "ModelConfig":
        return cls(cls.DEFAULT_BRAIN_HYBRID_CONFIG)

    # --- ACCESSOR METHODS ---
    
    def save_yaml(self, yaml_path: str):
        """
        Save configuration to YAML file.
        """
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w') as f:
            # Use settings.dict() to export validated Pydantic object
            yaml.dump(self.settings.model_dump(), f, default_flow_style=False)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value from the validated settings object.
        """
        # Access attributes directly from the Pydantic object
        return getattr(self.settings, key, default)
    
    def set(self, key: str, value: Any):
        """
        Set configuration value on the validated settings object.
        """
        # Pydantic objects are immutable by default, but we'll allow setting for overrides
        setattr(self.settings, key, value)
    
    def update(self, updates: Dict[str, Any]):
        """
        Update multiple configuration values.
        """
        # Pydantic update—requires creating a new instance or using private fields
        updated_dict = self.settings.model_dump()
        updated_dict.update(updates)
        self.settings = ModelSettings(**updated_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        """
        return self.settings.model_dump()
    
    def __repr__(self) -> str:
        return f"ModelConfig(settings={self.settings.model_dump_json()})"
    
    def __str__(self) -> str:
        """Pretty string representation."""
        lines = ["Model Configuration (Validated):"]
        # Use Pydantic's dict export for clear printing
        for key, value in self.settings.model_dump().items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)