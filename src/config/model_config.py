"""Model configuration management."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ModelConfig:
    """Configuration manager for model training and inference."""
    
    # Default configurations for ResNet-based models (kept for compatibility)
    DEFAULT_LUNG_CONFIG = {
        "model_name": "lung_cancer_classifier",
        "model_type": "LungCancerClassifier",
        "input_shape": [224, 224, 3],
        "num_classes": 2,
        "base_model": "ResNet50",
        "learning_rate": 0.0001,
        "batch_size": 32,
        "epochs": 50,
        "validation_split": 0.2,
        "freeze_base": True,
        "use_augmentation": True,
        "early_stopping_patience": 10,
        "reduce_lr_patience": 5,
        "class_labels": ["Normal", "Cancerous"],
    }
    
    DEFAULT_BRAIN_CONFIG = {
        "model_name": "brain_cancer_classifier",
        "model_type": "BrainCancerClassifier",
        "input_shape": [224, 224, 3],
        "num_classes": 4,
        "base_model": "ResNet50",
        "learning_rate": 0.0001,
        "batch_size": 32,
        "epochs": 50,
        "validation_split": 0.2,
        "freeze_base": True,
        "use_augmentation": True,
        "early_stopping_patience": 10,
        "reduce_lr_patience": 5,
        "class_labels": ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
    }

    # --- NEW HYBRID CONFIGURATIONS ---

    # Base settings for the Hybrid model (Brain Cancer specific)
    DEFAULT_BRAIN_HYBRID_CONFIG = {
        "model_name": "brain_effresnet_vit_classifier",
        "model_type": "EffResNetViTClassifier",
        "input_shape": [224, 224, 3],
        "num_classes": 4,
        "base_model": "EffResNet", # Triggers fusion logic in EffResNetViTClassifier
        "learning_rate": 0.00005,
        "batch_size": 16, 
        "epochs": 75,
        "validation_split": 0.2,
        "freeze_base": True,
        "use_augmentation": True,
        "early_stopping_patience": 12,
        "reduce_lr_patience": 6,
        "class_labels": ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
    }

    # Base settings for the Hybrid model (Lung Cancer specific)
    DEFAULT_LUNG_HYBRID_CONFIG = {
        "model_name": "lung_effresnet_vit_classifier",
        "model_type": "EffResNetViTClassifier",
        "input_shape": [224, 224, 3],
        "num_classes": 2,
        "base_model": "EffResNet", # Triggers fusion logic in EffResNetViTClassifier
        "learning_rate": 0.00005,
        "batch_size": 16, 
        "epochs": 75,
        "validation_split": 0.2,
        "freeze_base": True,
        "use_augmentation": True,
        "early_stopping_patience": 12,
        "reduce_lr_patience": 6,
        "class_labels": ["Normal", "Cancerous"],
    }

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration.
        
        Args:
            config_dict: Configuration dictionary
        """
        self.config = config_dict if config_dict else {}
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ModelConfig":
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            ModelConfig instance
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(config_dict)
    
    @classmethod
    def lung_cancer_default(cls) -> "ModelConfig":
        """
        Create default lung cancer (ResNet) configuration.
        """
        return cls(cls.DEFAULT_LUNG_CONFIG.copy())
    
    @classmethod
    def brain_cancer_default(cls) -> "ModelConfig":
        """
        Create default brain cancer (ResNet) configuration.
        """
        return cls(cls.DEFAULT_BRAIN_CONFIG.copy())

    # --- NEW HYBRID CLASS METHODS ---
    @classmethod
    def lung_hybrid_default(cls) -> "ModelConfig":
        """
        Create default hybrid EffResNet-ViT configuration for LUNG.
        """
        return cls(cls.DEFAULT_LUNG_HYBRID_CONFIG.copy())

    @classmethod
    def brain_hybrid_default(cls) -> "ModelConfig":
        """
        Create default hybrid EffResNet-ViT configuration for BRAIN.
        """
        return cls(cls.DEFAULT_BRAIN_HYBRID_CONFIG.copy())
    
    # [Rest of the methods: save_yaml, get, set, update, to_dict, __repr__, __str__]
    # (These were already provided and do not need modification)
    def save_yaml(self, yaml_path: str):
        """
        Save configuration to YAML file.
        
        Args:
            yaml_path: Path to save YAML configuration
        """
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """
        Set configuration value.
        """
        self.config[key] = value
    
    def update(self, updates: Dict[str, Any]):
        """
        Update multiple configuration values.
        """
        self.config.update(updates)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        """
        return self.config.copy()
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"ModelConfig({self.config})"
    
    def __str__(self) -> str:
        """Pretty string representation."""
        lines = ["Model Configuration:"]
        for key, value in self.config.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)