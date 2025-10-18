"""Example script for training lung cancer classification model."""

import os
import sys
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from numpy import unique

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Remove LungCancerClassifier import
from src.models import EffResNetViTClassifier
from src.data import DataLoader
from src.config import ModelConfig
from src.utils import Visualizer

def main():
    """Main training function."""
    print("=" * 60)
    print("Lung Cancer Classification Model Training (EffResNet-ViT)")
    print("=" * 60)
    
    # Load configuration
    # Use the specific hybrid default for lung cancer
    config = ModelConfig.lung_hybrid_default() # <--- CORRECT CONFIG CALL
    print("\nConfiguration:")
    print(config) 
    
    # Initialize data loader
    data_dir = config.get("data_dir", "data/lung_cancer/train")
    
    if not os.path.exists(data_dir):
        print(f"\nError: Data directory not found: {data_dir}")
        print("\nPlease organize your data in the following structure:")
        print("data/lung_cancer/train/")
        print("  ├── Normal/")
        print("  │   ├── image1.jpg")
        print("  │   └── image2.jpg")
        print("  └── Cancerous/")
        print("      ├── image1.jpg")
        print("      └── image2.jpg")
        return
    
    print(f"\nLoading data from: {data_dir}")
    
    data_loader = DataLoader(
        data_dir=data_dir,
        image_size=tuple(config.get("input_shape")[:2]),
        batch_size=config.get("batch_size"),
        validation_split=config.get("validation_split")
    )
    
    # Load training and validation data
    print("\nLoading training and validation datasets...")
    train_data, val_data = data_loader.load_train_validation_datasets(
        augment_train=config.get("use_augmentation")
    )
    
    print(f"Training samples: {train_data.samples}")
    print(f"Validation samples: {val_data.samples}")
    print(f"Classes: {train_data.class_indices}")

    # Calculate class weights for imbalance (Task 1)
    class_indices = list(train_data.class_indices.values())
    if train_data.samples > 0 and len(class_indices) > 0:
        class_weights_array = compute_class_weight(
            'balanced',
            classes=unique(class_indices),
            y=train_data.classes
        )
        class_weights_dict = dict(zip(unique(class_indices), class_weights_array))
        print(f"\nCalculated Class Weights: {class_weights_dict}")
    else:
        class_weights_dict = None
        print("\nCould not calculate class weights (No samples or classes found).")
    
    # Initialize model
    print("\nInitializing hybrid EffResNet-ViT model...")
    model = EffResNetViTClassifier(
        input_shape=tuple(config.get("input_shape")),
        num_classes=config.get("num_classes"),
        base_model_name=config.get("base_model"),
        learning_rate=config.get("learning_rate")
    )

    # Set the human-readable labels from config (Task 3)
    model.set_class_labels(config.get("class_labels"))
    
    
    # Build model
    print("\nBuilding model architecture...")
    model.build_model(freeze_base=config.get("freeze_base"))
    
    print("\nModel Information:")
    for key, value in model.get_model_info().items():
        print(f"  {key}: {value}")
    
    print("\nModel Summary:")
    model.model.summary()
    
    # Train model
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60)
    
    history = model.train(
        train_data=train_data,
        validation_data=val_data,
        epochs=config.get("epochs"),
        batch_size=config.get("batch_size"),
        class_weights=class_weights_dict # <--- PASSING WEIGHTS HERE
    )
    
    # Save model
    model_save_path = f"models/saved/{config.get('model_name')}.h5"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    print(f"\nSaving model to: {model_save_path}")
    model.save_model(model_save_path)
    
    # Visualize training history
    print("\nGenerating training history plots...")
    visualizer = Visualizer()
    visualizer.plot_training_history(
        history,
        metrics=['loss', 'accuracy', 'auc'],
        save_path=f"results/{config.get('model_name')}_training_history.png"
    )

    # Optional: Demonstrate Grad-CAM on a single validation image (Task 2)
    # print("\nDemonstrating Grad-CAM on a sample image (Requires trained weights)...")
    # try:
    #     sample_image = next(iter(val_data))[0][0] # Get first image from first batch
    #     # Last convolutional layer name set in EffResNetViTClassifier.build_model
    #     Visualizer.plot_grad_cam(model.model, sample_image, "last_cnn_features")
    # except Exception as e:
    #     print(f"Grad-CAM visualization failed: {e}")
    
    # Evaluate on validation data
    print("\nEvaluating on validation data...")
    val_metrics = model.evaluate(val_data)
    
    print("\nValidation Metrics:")
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()