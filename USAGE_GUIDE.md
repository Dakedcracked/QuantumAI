# Medical AI Usage Guide

## Overview

This guide provides detailed instructions on using the Medical AI system for disease detection from medical images.

## Table of Contents

1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Training a Model](#training-a-model)
4. [Making Predictions](#making-predictions)
5. [Model Evaluation](#model-evaluation)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-enabled GPU for faster training

### Step 1: Clone the Repository

```bash
git clone https://github.com/Dakedcracked/QuantumAI.git
cd QuantumAI
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** TensorFlow installation may take some time. For GPU support, install `tensorflow-gpu` instead.

## Dataset Preparation

### Directory Structure

Organize your medical images in the following structure:

```
dataset/
├── normal/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── pneumonia/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── covid19/
    ├── image001.jpg
    ├── image002.jpg
    └── ...
```

### Image Requirements

- **Format:** JPG, PNG, or other common image formats
- **Size:** Any size (will be automatically resized to 224x224)
- **Color:** RGB or grayscale images
- **Naming:** Any naming convention is acceptable

### Dataset Splitting

The system automatically splits your data into:
- **Training set:** 80% of images
- **Validation set:** 20% of images

## Training a Model

### Using Jupyter Notebook

1. Open `model.ipynb` in Jupyter Notebook or JupyterLab
2. Run the import cell to load all dependencies
3. Update the dataset path in the usage example
4. Execute cells sequentially to train the model

### Using Python Script

```python
from medical_ai import MedicalImageDataLoader, MedicalAIModel, ModelEvaluator

# Configuration
DATA_PATH = 'path/to/dataset'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50

# Load data
data_loader = MedicalImageDataLoader(image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
train_data, val_data = data_loader.load_data(DATA_PATH)

# Build model
num_classes = len(train_data.class_indices)
medical_ai = MedicalAIModel(num_classes=num_classes)
medical_ai.build_model()
medical_ai.compile_model(learning_rate=0.001)

# Train
history = medical_ai.train(train_data, val_data, epochs=EPOCHS)

# Save
medical_ai.save_model('my_medical_model.h5')
```

### Training Parameters

- **Image Size:** Default (224, 224) - standard for medical imaging
- **Batch Size:** 32 (adjust based on GPU memory)
- **Epochs:** 50 (will stop early if validation loss doesn't improve)
- **Learning Rate:** 0.001 (automatically reduced during training)

## Making Predictions

### Single Image Prediction

```python
from medical_ai import MedicalImageDataLoader, MedicalAIModel
import numpy as np

# Load model
medical_ai = MedicalAIModel(num_classes=3)
medical_ai.load_model('my_medical_model.h5')

# Preprocess image
data_loader = MedicalImageDataLoader()
image = data_loader.preprocess_single_image('test_image.jpg')

# Predict
prediction = medical_ai.predict(image)
class_names = ['Normal', 'Pneumonia', 'COVID-19']
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print(f"Predicted: {predicted_class} ({confidence:.2f}%)")
```

### Batch Predictions

```python
test_images = ['image1.jpg', 'image2.jpg', 'image3.jpg']

for image_path in test_images:
    image = data_loader.preprocess_single_image(image_path)
    prediction = medical_ai.predict(image)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    print(f"{image_path}: {predicted_class} ({confidence:.2f}%)")
```

## Model Evaluation

### Visualize Training History

```python
from medical_ai import ModelEvaluator

class_names = ['Normal', 'Pneumonia', 'COVID-19']
evaluator = ModelEvaluator(medical_ai.model, class_names)

# Plot training metrics
evaluator.plot_training_history(history)
```

This generates a plot with:
- Training and validation accuracy
- Training and validation loss
- AUC scores
- Precision scores

### Confusion Matrix

```python
# Evaluate on validation data
y_true, y_pred = evaluator.evaluate_model(val_data)

# Plot confusion matrix
evaluator.plot_confusion_matrix(y_true, y_pred)
```

### Classification Report

```python
# Generate detailed classification report
evaluator.generate_classification_report(y_true, y_pred)
```

Output includes:
- Precision
- Recall
- F1-score
- Support for each class

## Advanced Usage

### Custom Model Architecture

```python
# Build model with custom parameters
medical_ai = MedicalAIModel(
    num_classes=5,
    input_shape=(256, 256, 3)  # Custom image size
)
model = medical_ai.build_model()
```

### Adjust Training Callbacks

```python
import tensorflow as tf

# Custom callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,  # Wait 15 epochs
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,  # Reduce LR by 80%
        patience=7,
        min_lr=1e-8
    )
]

history = medical_ai.model.fit(
    train_data,
    validation_data=val_data,
    epochs=100,
    callbacks=callbacks
)
```

### Data Augmentation Configuration

```python
# Custom data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,      # Rotate images
    width_shift_range=0.3,  # Horizontal shift
    height_shift_range=0.3, # Vertical shift
    horizontal_flip=True,
    vertical_flip=True,     # For medical images
    zoom_range=0.2,         # Zoom in/out
    fill_mode='nearest'
)
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory Error

**Problem:** GPU runs out of memory during training

**Solution:**
```python
# Reduce batch size
data_loader = MedicalImageDataLoader(batch_size=16)  # or 8
```

#### 2. Low Accuracy

**Problem:** Model accuracy is low

**Solutions:**
- Increase training epochs
- Add more training data
- Use data augmentation
- Try different learning rates
- Check data quality and labeling

#### 3. Overfitting

**Problem:** Training accuracy is high, but validation accuracy is low

**Solutions:**
- Increase dropout rates
- Add more data augmentation
- Reduce model complexity
- Add early stopping

#### 4. Installation Issues

**Problem:** TensorFlow installation fails

**Solution:**
```bash
# For CPU-only
pip install tensorflow-cpu

# For GPU support (requires CUDA)
pip install tensorflow-gpu
```

### Performance Tips

1. **Use GPU:** Significantly faster training
2. **Batch Size:** Larger batches = faster training (if GPU memory allows)
3. **Image Size:** Smaller images = faster training (but may reduce accuracy)
4. **Mixed Precision:** Enable for faster training on compatible GPUs

```python
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

## Best Practices

1. **Data Quality:** Ensure consistent image quality and proper labeling
2. **Class Balance:** Try to have similar numbers of images per class
3. **Validation:** Always evaluate on a separate test set
4. **Documentation:** Keep track of training parameters and results
5. **Version Control:** Save models with version numbers or dates
6. **Privacy:** Ensure compliance with medical data regulations (HIPAA, GDPR)

## Support

For issues or questions:
- Check the [README.md](README.md) file
- Review the example code in `example_usage.py`
- Open an issue on GitHub

## License

This project is open source and available for research and educational purposes.
