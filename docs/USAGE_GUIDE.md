# QuantumAI Usage Guide

This comprehensive guide will walk you through using the QuantumAI medical image classification system.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Data Preparation](#data-preparation)
3. [Training Models](#training-models)
4. [Making Predictions](#making-predictions)
5. [Model Evaluation](#model-evaluation)
6. [Advanced Techniques](#advanced-techniques)
7. [Troubleshooting](#troubleshooting)

## Getting Started

### Installation

```bash
# Clone and install
git clone https://github.com/Dakedcracked/QuantumAI.git
cd QuantumAI
pip install -r requirements.txt
```

### Verify Installation

```python
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
```

## Data Preparation

### Organizing Your Dataset

#### For Lung Cancer Detection

Create the following structure:

```
data/lung_cancer/
├── train/
│   ├── Normal/
│   │   ├── normal_001.jpg
│   │   ├── normal_002.jpg
│   │   └── ...
│   └── Cancerous/
│       ├── cancer_001.jpg
│       ├── cancer_002.jpg
│       └── ...
```

#### For Brain Tumor Detection

```
data/brain_cancer/
├── train/
│   ├── Glioma/
│   ├── Meningioma/
│   ├── No_Tumor/
│   └── Pituitary/
```

### Image Requirements

- **Format**: JPEG, PNG, BMP, or TIFF
- **Size**: Minimum 224x224 pixels (will be resized)
- **Quality**: High-resolution medical scans recommended
- **Color**: RGB or Grayscale (automatically converted)

### Data Quality Tips

1. **Remove duplicates**: Ensure no duplicate images
2. **Balance classes**: Try to have similar numbers of images per class
3. **Quality check**: Remove corrupted or low-quality images
4. **Anonymize**: Remove patient information from images

## Training Models

### Basic Training - Lung Cancer

```python
from src.models import LungCancerClassifier
from src.data import DataLoader

# Initialize data loader
data_loader = DataLoader(
    data_dir="data/lung_cancer/train",
    image_size=(224, 224),
    batch_size=32,
    validation_split=0.2
)

# Load data
train_data, val_data = data_loader.load_train_validation_datasets(
    augment_train=True
)

# Initialize model
model = LungCancerClassifier(
    input_shape=(224, 224, 3),
    num_classes=2,
    base_model_name="ResNet50",
    learning_rate=0.0001
)

# Build and train
model.build_model(freeze_base=True)
history = model.train(
    train_data=train_data,
    validation_data=val_data,
    epochs=50,
    batch_size=32
)

# Save model
model.save_model("models/saved/lung_cancer_model.h5")
```

## Making Predictions

### Single Image Prediction

```python
from src.models import LungCancerClassifier
from src.utils import ImagePreprocessor
import numpy as np

# Load model
model = LungCancerClassifier()
model.load_model("models/saved/lung_cancer_model.h5")

# Prepare image
preprocessor = ImagePreprocessor(target_size=(224, 224))
image = preprocessor.load_and_preprocess_image("scan.jpg")

# Predict
predictions, labels = model.predict_with_labels(np.array([image]))

print(f"Prediction: {labels[0]}")
print(f"Confidence: {predictions[0][0]:.2%}")
```

## Model Evaluation

### Comprehensive Evaluation

```python
from src.utils import ModelEvaluator
import numpy as np

# Load test data
test_loader = DataLoader(data_dir="data/lung_cancer/test")
test_data = test_loader.load_dataset(subset='training', shuffle=False)

# Get predictions
y_true = test_data.classes
y_pred_proba = model.predict(test_data)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Evaluate
evaluator = ModelEvaluator(class_labels=["Normal", "Cancerous"])
metrics = evaluator.evaluate_binary_classification(y_true, y_pred_proba)

print("Evaluation Metrics:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory Error

**Solution**: Reduce batch size

```python
data_loader = DataLoader(
    data_dir="data/lung_cancer/train",
    batch_size=8  # Reduced from 32
)
```

#### 2. Model Not Converging

**Solutions**:
- Reduce learning rate
- Use data augmentation
- Check data quality
- Increase training epochs

For more information, refer to the API documentation and example scripts.
