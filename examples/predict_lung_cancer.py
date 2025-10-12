"""Example script for lung cancer prediction on new images."""

import os
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import LungCancerClassifier
from src.utils import ImagePreprocessor, Visualizer

def main():
    """Main prediction function."""
    print("=" * 60)
    print("Lung Cancer Prediction")
    print("=" * 60)
    
    # Check if model exists
    model_path = "models/saved/lung_cancer_classifier.h5"
    
    if not os.path.exists(model_path):
        print(f"\nError: Model not found at {model_path}")
        print("Please train the model first using train_lung_cancer_model.py")
        return
    
    # Initialize model and load weights
    print(f"\nLoading model from: {model_path}")
    model = LungCancerClassifier(
        input_shape=(224, 224, 3),
        num_classes=2,
        base_model_name="ResNet50"
    )
    model.load_model(model_path)
    
    print("\nModel Information:")
    for key, value in model.get_model_info().items():
        print(f"  {key}: {value}")
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(
        target_size=(224, 224),
        normalize=True,
        clahe=False
    )
    
    # Example: Predict on images from a directory
    test_dir = "data/lung_cancer/test"
    
    if not os.path.exists(test_dir):
        print(f"\nTest directory not found: {test_dir}")
        print("Please provide images in the test directory for prediction")
        return
    
    print(f"\nLoading images from: {test_dir}")
    
    # Get all image files
    image_paths = []
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    for ext in valid_extensions:
        image_paths.extend(Path(test_dir).rglob(f"*{ext}"))
        image_paths.extend(Path(test_dir).rglob(f"*{ext.upper()}"))
    
    if not image_paths:
        print("No images found in test directory")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Load and preprocess images
    images = []
    filenames = []
    
    for img_path in image_paths[:10]:  # Process first 10 images
        try:
            img = preprocessor.load_and_preprocess_image(str(img_path))
            images.append(img)
            filenames.append(img_path.name)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    if not images:
        print("No images could be loaded")
        return
    
    images = np.array(images)
    
    # Make predictions
    print(f"\nMaking predictions on {len(images)} images...")
    predictions, predicted_labels = model.predict_with_labels(images)
    
    # Display results
    print("\n" + "=" * 60)
    print("Prediction Results:")
    print("=" * 60)
    
    for i, (filename, pred, label) in enumerate(zip(filenames, predictions, predicted_labels)):
        confidence = pred[0] if len(pred.shape) == 2 else pred
        if label == "Normal":
            confidence = 1 - confidence
        
        print(f"\n{i+1}. {filename}")
        print(f"   Prediction: {label}")
        print(f"   Confidence: {confidence:.2%}")
    
    # Visualize results
    print("\nGenerating visualization...")
    visualizer = Visualizer()
    
    # Denormalize images for visualization
    display_images = [preprocessor.denormalize_image(img) for img in images]
    
    visualizer.plot_images(
        display_images,
        predictions=predicted_labels,
        cols=4,
        save_path="results/lung_cancer_predictions.png"
    )
    
    print("\n" + "=" * 60)
    print("Prediction Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
