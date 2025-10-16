"""Model configuration management."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ModelConfig:
    """Configuration manager for model training and inference."""
    
    # Default configurations
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

    DEFAULT_EFFRESNET_VIT_CONFIG = {
        "model_name": "effresnet_vit_classifier",
        "model_type": "EffResNetViTClassifier",
        "input_shape": [224, 224, 3],
        "num_classes": 2,
        "base_model": "EffResNet",
        "learning_rate": 0.0001,
        "batch_size": 16,
        "epochs": 50,
        "validation_split": 0.2,
        "freeze_base": True,
        "use_augmentation": True,
        "early_stopping_patience": 10,
        "reduce_lr_patience": 5,
        "class_labels": ["Negative", "Positive"],
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
        Create default lung cancer configuration.
        
        Returns:
            ModelConfig instance with lung cancer defaults
        """
        return cls(cls.DEFAULT_LUNG_CONFIG.copy())
    
    @classmethod
    def brain_cancer_default(cls) -> "ModelConfig":
        """
        Create default brain cancer configuration.
        
        Returns:
            ModelConfig instance with brain cancer defaults
        """
        return cls(cls.DEFAULT_BRAIN_CONFIG.copy())

    @classmethod
    def effresnet_vit_default(cls) -> "ModelConfig":
        """
        Create default EffResNet-ViT configuration.

        Returns:
            ModelConfig instance with effresnet_vit defaults
        """
        return cls(cls.DEFAULT_EFFRESNET_VIT_CONFIG.copy())
    
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
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Value to set
        """
        self.config[key] = value
    
    def update(self, updates: Dict[str, Any]):
        """
        Update multiple configuration values.
        
        Args:
            updates: Dictionary of updates
        """
        self.config.update(updates)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration dictionary
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
