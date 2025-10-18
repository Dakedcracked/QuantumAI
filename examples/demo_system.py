"""Comprehensive demo of the QuantumAI system."""

import os
import numpy as np
import sys
from pathlib import Path
import tensorflow as tf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# --- FIX: Only import the currently active and necessary models/modules ---
from src.models import EffResNetViTClassifier # This is the active model class
from src.config import ModelConfig
from src.utils import ImagePreprocessor, ModelEvaluator, Visualizer


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def demo_lung_cancer_classifier():
    """Demonstrate the LUNG hybrid classifier."""
    print_section("LUNG CANCER CLASSIFICATION SYSTEM (EffResNet-ViT)")
    
    # Load the LUNG HYBRID configuration
    config = ModelConfig.lung_hybrid_default()
    
    print("Configuration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    
    # Initialize model
    print("\nInitializing EffResNet-ViT Classifier (LUNG)...")
    
    model = EffResNetViTClassifier(
        input_shape=tuple(config.get("input_shape")),
        num_classes=config.get("num_classes"),
        base_model_name=config.get("base_model"), # Should be "EffResNet"
        learning_rate=config.get("learning_rate")
    )
    # Set labels for the get_model_info call
    model.set_class_labels(config.get("class_labels"))
    
    # Build model
    print("\nBuilding model architecture...")
    model.build_model(freeze_base=config.get("freeze_base"))
    
    # Display model info
    print("\nModel Information:")
    for key, value in model.get_model_info().items():
        print(f"  {key}: {value}")
    
    # Display model summary (simplified parameter count for demo)
    print("\nModel Architecture Summary:")
    print(f"  Total parameters: {model.model.count_params():,}")
    print("\n✓ Lung Cancer Classifier initialized successfully!")
    
    return model


def demo_brain_cancer_classifier():
    """Demonstrate the BRAIN hybrid classifier."""
    print_section("BRAIN CANCER CLASSIFICATION SYSTEM (EffResNet-ViT)")
    
    # Load the BRAIN HYBRID configuration
    config = ModelConfig.brain_hybrid_default()
    
    print("Configuration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    
    # Initialize model
    print("\nInitializing EffResNet-ViT Classifier (BRAIN)...")
    
    model = EffResNetViTClassifier(
        input_shape=tuple(config.get("input_shape")),
        num_classes=config.get("num_classes"),
        base_model_name=config.get("base_model"), # Should be "EffResNet"
        learning_rate=config.get("learning_rate")
    )
    # Set labels for the get_model_info call
    model.set_class_labels(config.get("class_labels"))

    # Build model
    print("\nBuilding model architecture...")
    model.build_model(freeze_base=config.get("freeze_base"))
    
    # Display model info
    print("\nModel Information:")
    for key, value in model.get_model_info().items():
        print(f"  {key}: {value}")
    
    # Display model summary (simplified parameter count for demo)
    print("\nModel Architecture Summary:")
    print(f"  Total parameters: {model.model.count_params():,}")
    
    print("\n✓ Brain Cancer Classifier initialized successfully!")
    
    return model


def demo_preprocessing():
    """Demonstrate image preprocessing."""
    print_section("IMAGE PREPROCESSING UTILITIES")
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(
        target_size=(224, 224),
        normalize=True,
        clahe=True
    )
    
    print("Preprocessor Configuration:")
    print(f"  Target size: {preprocessor.target_size}")
    print(f"  Normalize: {preprocessor.normalize}")
    print(f"  CLAHE: {preprocessor.clahe}")
    
    print("\nPreprocessing capabilities:")
    print("  ✓ Image resizing")
    print("  ✓ Normalization")
    print("  ✓ CLAHE contrast enhancement")
    print("  ✓ Noise removal (Bilateral, Gaussian, Median)")
    print("  ✓ Histogram equalization")
    print("  ✓ Grayscale to RGB conversion")
    
    print("\n✓ Preprocessing utilities initialized successfully!")

def demo_evaluation():
    """Demonstrate model evaluation."""
    print_section("MODEL EVALUATION UTILITIES")
    
    # Labels are for demonstration purposes only
    evaluator = ModelEvaluator(class_labels=["Normal", "Cancerous"])
    
    print("Evaluation Metrics Supported:")
    print("  ✓ Accuracy")
    print("  ✓ Precision")
    print("  ✓ Recall (Sensitivity)")
    print("  ✓ F1-Score")
    print("  ✓ Specificity")
    print("  ✓ AUC-ROC")
    print("  ✓ Confusion Matrix")
    print("  ✓ Classification Report")
    
    print("\n✓ Evaluation utilities initialized successfully!")

def demo_visualization():
    """Demonstrate visualization utilities."""
    print_section("VISUALIZATION UTILITIES")
    
    visualizer = Visualizer(figsize=(12, 8))
    
    print("Visualization Capabilities:")
    print("  ✓ Image grid display")
    print("  ✓ Training history plots")
    print("  ✓ Confusion matrix heatmap")
    print("  ✓ ROC curve")
    print("  ✓ Prediction distribution")
    print("  ✓ Model comparison charts")
    print("  ✓ Grad-CAM (Interpretability hooks are available)")
    
    print("\n✓ Visualization utilities initialized successfully!")

def demo_config_management():
    """Demonstrate configuration management."""
    print_section("CONFIGURATION MANAGEMENT")
    
    # Create custom configuration
    custom_config = ModelConfig({
        "model_name": "custom_classifier",
        "num_classes": 2,
        "base_model": "DenseNet121",
        "class_labels": ["A", "B"],
        "learning_rate": 0.00005,
        "epochs": 100,
    })
    
    print("Configuration Management Features:")
    print("  ✓ YAML file support (with Pydantic Validation)")
    print("  ✓ Default configurations (Task-specific Hybrid)")
    print("  ✓ Custom configurations")
    print("  ✓ Easy parameter updates")
    
    # Save configuration
    config_path = "configs/demo_config.yaml"
    custom_config.save_yaml(config_path)
    print(f"\n✓ Configuration saved to: {config_path}")
    
    # Load configuration
    loaded_config = ModelConfig.from_yaml(config_path)
    print("✓ Configuration loaded and validated successfully!")
    
    return custom_config

def main():
    """Main demo function."""
    print("\n" + "=" * 70)
    print("  QUANTUMAI - MEDICAL IMAGE CLASSIFICATION SYSTEM")
    print("  Professional-Grade Cancer Detection & Classification (Hybrid ViT)")
    print("=" * 70)
    
    print("\nThis demonstration showcases the capabilities of QuantumAI:")
    print("  • Lung Cancer Detection (via EffResNet-ViT)")
    print("  • Brain Tumor Classification (via EffResNet-ViT)")
    print("  • Advanced preprocessing and augmentation")
    print("  • Comprehensive evaluation metrics")
    print("  • Professional visualization tools")
    
    try:
        # Import TensorFlow to check availability
        import tensorflow as tf
        print(f"\n✓ TensorFlow {tf.__version__} detected")
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU detected: {len(gpus)} device(s)")
        else:
            print("✓ Running on CPU")
        
        # Demo each component
        demo_lung_cancer_classifier()
        demo_brain_cancer_classifier()
        demo_preprocessing()
        demo_evaluation()
        demo_visualization()
        demo_config_management()
        
        # Final summary
        print_section("DEMONSTRATION COMPLETE")
        
        print("System Components Ready:")
        print("  ✓ Hybrid EffResNet-ViT Classifier")
        print("  ✓ Image Preprocessing")
        print("  ✓ Model Evaluation")
        print("  ✓ Visualization Tools")
        print("  ✓ Configuration Management")
        
        print("\nNext Steps:")
        print("  1. Prepare your medical image datasets")
        print("  2. Run training scripts:")
        print("     - python examples/train_lung_cancer_model.py")
        print("     - python examples/train_brain_cancer_model.py")
        print("  3. Make predictions on new images:")
        print("     - python examples/predict_lung_cancer.py")
        print("     - python examples/predict_brain_cancer.py")
        
        print("\n" + "=" * 70)
        print("  Thank you for using QuantumAI!")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error during demonstration: {e}")
        print("Please ensure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)