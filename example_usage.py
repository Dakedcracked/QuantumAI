"""
Example usage of the Medical AI system for disease detection.
This script demonstrates how to use the MedicalAI framework.
"""

from medical_ai import MedicalImageDataLoader, MedicalAIModel, ModelEvaluator
import numpy as np


def train_model_example():
    """
    Example: Training a medical AI model for disease detection.
    """
    print("=" * 70)
    print("MEDICAL AI - TRAINING EXAMPLE")
    print("=" * 70)
    
    # Configuration
    DATA_PATH = 'path/to/dataset'  # Update with your dataset path
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    print("\nStep 1: Loading and preprocessing data...")
    print(f"  - Image size: {IMAGE_SIZE}")
    print(f"  - Batch size: {BATCH_SIZE}")
    
    # Initialize data loader
    data_loader = MedicalImageDataLoader(image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
    
    # Load data (uncomment when you have dataset)
    # train_data, val_data = data_loader.load_data(DATA_PATH)
    # print(f"  - Training samples: {train_data.samples}")
    # print(f"  - Validation samples: {val_data.samples}")
    # print(f"  - Classes: {list(train_data.class_indices.keys())}")
    
    print("\nStep 2: Building the model...")
    # num_classes = len(train_data.class_indices)
    num_classes = 3  # Example: Normal, Pneumonia, COVID-19
    medical_ai = MedicalAIModel(num_classes=num_classes, input_shape=(224, 224, 3))
    model = medical_ai.build_model()
    print(f"  - Model created with {num_classes} classes")
    
    # Display model summary
    print("\nModel Architecture Summary:")
    model.summary()
    
    print("\nStep 3: Compiling the model...")
    medical_ai.compile_model(learning_rate=LEARNING_RATE)
    
    print("\nStep 4: Training the model...")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print("  - Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint")
    
    # Train model (uncomment when you have dataset)
    # history = medical_ai.train(train_data, val_data, epochs=EPOCHS)
    
    print("\nStep 5: Evaluating the model...")
    # class_names = list(train_data.class_indices.keys())
    # evaluator = ModelEvaluator(medical_ai.model, class_names)
    # evaluator.plot_training_history(history)
    
    # y_true, y_pred = evaluator.evaluate_model(val_data)
    # evaluator.plot_confusion_matrix(y_true, y_pred)
    # evaluator.generate_classification_report(y_true, y_pred)
    
    print("\nStep 6: Saving the model...")
    # medical_ai.save_model('medical_ai_model.h5')
    
    print("\n" + "=" * 70)
    print("Training complete! Model saved.")
    print("=" * 70)


def predict_single_image_example():
    """
    Example: Making predictions on a single medical image.
    """
    print("\n" + "=" * 70)
    print("MEDICAL AI - PREDICTION EXAMPLE")
    print("=" * 70)
    
    # Configuration
    MODEL_PATH = 'medical_ai_model.h5'
    IMAGE_PATH = 'path/to/test/image.jpg'
    IMAGE_SIZE = (224, 224)
    
    print("\nStep 1: Loading the model...")
    # medical_ai = MedicalAIModel(num_classes=3)  # Update with your number of classes
    # medical_ai.load_model(MODEL_PATH)
    
    print("\nStep 2: Preprocessing the image...")
    data_loader = MedicalImageDataLoader(image_size=IMAGE_SIZE)
    # image = data_loader.preprocess_single_image(IMAGE_PATH)
    
    print("\nStep 3: Making prediction...")
    # prediction = medical_ai.predict(image)
    
    # Example class names
    class_names = ['Normal', 'Pneumonia', 'COVID-19']
    
    # Get predicted class and confidence (uncomment when you have model and image)
    # predicted_class_idx = np.argmax(prediction)
    # predicted_class = class_names[predicted_class_idx]
    # confidence = np.max(prediction) * 100
    
    # Print results
    # print("\nPrediction Results:")
    # print("-" * 70)
    # print(f"Image: {IMAGE_PATH}")
    # print(f"Predicted Disease: {predicted_class}")
    # print(f"Confidence: {confidence:.2f}%")
    # print("\nClass Probabilities:")
    # for i, class_name in enumerate(class_names):
    #     prob = prediction[0][i] * 100
    #     print(f"  {class_name}: {prob:.2f}%")
    
    print("\n" + "=" * 70)
    print("Prediction complete!")
    print("=" * 70)


def batch_prediction_example():
    """
    Example: Making predictions on multiple medical images.
    """
    print("\n" + "=" * 70)
    print("MEDICAL AI - BATCH PREDICTION EXAMPLE")
    print("=" * 70)
    
    # Configuration
    MODEL_PATH = 'medical_ai_model.h5'
    TEST_IMAGES = [
        'path/to/test/image1.jpg',
        'path/to/test/image2.jpg',
        'path/to/test/image3.jpg'
    ]
    IMAGE_SIZE = (224, 224)
    class_names = ['Normal', 'Pneumonia', 'COVID-19']
    
    print("\nProcessing multiple images...")
    print(f"Total images: {len(TEST_IMAGES)}")
    
    # Load model
    # medical_ai = MedicalAIModel(num_classes=len(class_names))
    # medical_ai.load_model(MODEL_PATH)
    
    # Initialize data loader
    data_loader = MedicalImageDataLoader(image_size=IMAGE_SIZE)
    
    # Process each image
    # for idx, image_path in enumerate(TEST_IMAGES, 1):
    #     print(f"\nImage {idx}: {image_path}")
    #     image = data_loader.preprocess_single_image(image_path)
    #     prediction = medical_ai.predict(image)
    #     predicted_class = class_names[np.argmax(prediction)]
    #     confidence = np.max(prediction) * 100
    #     print(f"  Predicted: {predicted_class} ({confidence:.2f}%)")
    
    print("\n" + "=" * 70)
    print("Batch prediction complete!")
    print("=" * 70)


def main():
    """
    Main function to demonstrate all examples.
    """
    print("\n")
    print("*" * 70)
    print("MEDICAL AI - COMPREHENSIVE USAGE EXAMPLES")
    print("*" * 70)
    print("\nThis script demonstrates how to use the Medical AI framework.")
    print("Uncomment the relevant sections and update paths to run the examples.")
    print("\nIMPORTANT:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Organize your dataset in the required structure")
    print("3. Update file paths in the examples")
    print("\nDataset structure should be:")
    print("  dataset/")
    print("    class1/")
    print("      image1.jpg")
    print("      image2.jpg")
    print("    class2/")
    print("      image1.jpg")
    print("      image2.jpg")
    print("*" * 70)
    
    # Run examples
    train_model_example()
    predict_single_image_example()
    batch_prediction_example()
    
    print("\n" + "*" * 70)
    print("For more information, see README.md")
    print("*" * 70 + "\n")


if __name__ == "__main__":
    main()
