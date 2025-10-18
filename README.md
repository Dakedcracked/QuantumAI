# QuantumAI - Medical Image Classification System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.10+](https://img.shields.io/badge/tensorflow-2.10+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A professional-grade deep learning system for medical image classification, specializing in **lung cancer** and **brain tumor/cancer** detection from medical imaging scans.

## 🎯 Features

- **Lung Cancer Classification**: Detect and classify lung cancer from CT scans and X-rays
- **Brain Tumor Classification**: Identify brain tumors (Glioma, Meningioma, Pituitary) from MRI scans
- **Transfer Learning**: Leverages state-of-the-art CNN architectures (ResNet50, VGG16, InceptionV3, DenseNet121, EfficientNet,EFFResNet-ViT)
- **Data Augmentation**: Advanced augmentation techniques for medical images
- **Professional Metrics**: Comprehensive evaluation with accuracy, precision, recall, F1-score, AUC-ROC, and specificity
- **Visualization Tools**: Built-in visualization for predictions, training history, and confusion matrices
- **Easy-to-Use API**: Simple and intuitive interface for training and inference

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Preparation](#dataset-preparation)
- [Training Models](#training-models)
- [Making Predictions](#making-predictions)
- [Model Architecture](#model-architecture)
- [Advanced Usage](#advanced-usage)
- [API Reference](#api-reference)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM

### Install from Source

```bash
# Clone the repository
git clone https://github.com/Dakedcracked/QuantumAI.git
cd QuantumAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## ⚡ Quick Start

### Training a Lung Cancer Classifier

```python
from src.models import LungCancerClassifier
from src.data import DataLoader
from src.config import ModelConfig

# Load configuration
config = ModelConfig.lung_cancer_default()

# Initialize data loader
data_loader = DataLoader(
    data_dir="data/lung_cancer/train",
    image_size=(224, 224),
    batch_size=32
)

# Load data
train_data, val_data = data_loader.load_train_validation_datasets()

# Initialize and train model
model = LungCancerClassifier(
    input_shape=(224, 224, 3),
    num_classes=2,
    base_model_name="ResNet50"
)

model.build_model()
model.train(train_data, val_data, epochs=50)

# Save model
model.save_model("models/saved/lung_cancer_classifier.h5")
```

### Making Predictions

```python
from src.models import LungCancerClassifier
from src.utils import ImagePreprocessor
import numpy as np

# Load model
model = LungCancerClassifier()
model.load_model("models/saved/lung_cancer_classifier.h5")

# Preprocess image
preprocessor = ImagePreprocessor(target_size=(224, 224))
image = preprocessor.load_and_preprocess_image("path/to/scan.jpg")

# Predict
predictions, labels = model.predict_with_labels(np.array([image]))
print(f"Prediction: {labels[0]}")
```

## 📁 Dataset Preparation

### Lung Cancer Dataset Structure

```
data/lung_cancer/
├── train/
│   ├── Normal/
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │   └── ...
│   └── Cancerous/
│       ├── image001.jpg
│       ├── image002.jpg
│       └── ...
├── test/
│   ├── Normal/
│   └── Cancerous/
└── validation/
    ├── Normal/
    └── Cancerous/
```

### Brain Cancer Dataset Structure

```
data/brain_cancer/
├── train/
│   ├── Glioma/
│   ├── Meningioma/
│   ├── No_Tumor/
│   └── Pituitary/
├── test/
└── validation/
```

### Supported Image Formats

- JPEG/JPG
- PNG
- BMP
- TIFF/TIF

### Image Requirements

- **Resolution**: Recommended minimum 224x224 pixels
- **Color Space**: RGB or Grayscale (automatically converted)
- **Quality**: High-resolution medical scans preferred

## 🎓 Training Models

### Using Command-Line Scripts

#### Lung Cancer Model

```bash
python examples/train_lung_cancer_model.py
```

#### Brain Cancer Model

```bash
python examples/train_brain_cancer_model.py
```

### Custom Training Configuration

```python
from src.config import ModelConfig

# Create custom configuration
config = ModelConfig({
    "model_name": "custom_lung_classifier",
    "input_shape": [224, 224, 3],
    "num_classes": 3,  # Normal, Benign, Malignant
    "base_model": "DenseNet121",
    "learning_rate": 0.0001,
    "batch_size": 16,
    "epochs": 100,
    "use_augmentation": True,
})

# Save configuration
config.save_yaml("configs/custom_config.yaml")
```

## 🔮 Making Predictions

### Batch Prediction

```bash
# Lung cancer prediction
python examples/predict_lung_cancer.py

# Brain cancer prediction
python examples/predict_brain_cancer.py
```

### Single Image Prediction

```python
from src.models import BrainCancerClassifier
from src.utils import ImagePreprocessor
import numpy as np

# Load model
model = BrainCancerClassifier()
model.load_model("models/saved/brain_cancer_classifier.h5")

# Preprocess
preprocessor = ImagePreprocessor()
image = preprocessor.load_and_preprocess_image("mri_scan.jpg")

# Predict
predictions, labels = model.predict_with_labels(np.array([image]))

print(f"Diagnosis: {labels[0]}")
print(f"Confidence: {np.max(predictions[0]):.2%}")
```

## 🏗️ Model Architecture

### Base Architectures Supported

- **ResNet50** (Default): Deep residual learning, excellent for medical imaging
- **VGG16**: Classic architecture, good baseline performance
- **InceptionV3**: Multi-scale feature extraction
- **DenseNet121**: Dense connections, efficient parameter usage
- **EfficientNetB0**: State-of-the-art efficiency and accuracy

### Model Components

1. **Base CNN**: Pre-trained on ImageNet for transfer learning
2. **Global Average Pooling**: Reduces spatial dimensions
3. **Dense Layers**: 512 → 256 neurons with BatchNorm and Dropout
4. **Output Layer**: Binary (sigmoid) or Multi-class (softmax)

### Training Features

- **Early Stopping**: Prevents overfitting
- **Learning Rate Reduction**: Adaptive learning rate scheduling
- **Data Augmentation**: Rotation, flip, zoom, shift, brightness, contrast
- **Class Balancing**: Handles imbalanced datasets
- **Regularization**: Dropout and BatchNormalization

## 🔧 Advanced Usage

### Custom Data Augmentation

```python
from src.data import DataAugmentor
import cv2

augmentor = DataAugmentor(seed=42)

# Load image
image = cv2.imread("scan.jpg")

# Apply augmentation
augmented = augmentor.augment(
    image,
    apply_rotation=True,
    apply_flip=True,
    apply_zoom=True,
    apply_brightness=True
)
```

### Model Evaluation

```python
from src.utils import ModelEvaluator
import numpy as np

evaluator = ModelEvaluator(class_labels=["Normal", "Cancerous"])

# Evaluate
metrics = evaluator.evaluate_binary_classification(y_true, y_pred)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1-Score: {metrics['f1_score']:.4f}")
print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
```

### Visualization

```python
from src.utils import Visualizer

visualizer = Visualizer()

# Plot training history
visualizer.plot_training_history(history, metrics=['loss', 'accuracy'])

# Plot confusion matrix
visualizer.plot_confusion_matrix(cm, class_labels=["Normal", "Cancerous"])

# Plot ROC curve
visualizer.plot_roc_curve(fpr, tpr, auc_score=0.95)
```

## 📊 API Reference

### LungCancerClassifier

```python
LungCancerClassifier(
    input_shape=(224, 224, 3),
    num_classes=2,
    base_model_name="ResNet50",
    learning_rate=0.0001
)
```

**Methods:**
- `build_model(freeze_base=True)`: Build model architecture
- `train(train_data, validation_data, epochs, batch_size)`: Train model
- `predict(images)`: Make predictions
- `predict_with_labels(images)`: Predict with class labels
- `evaluate(test_data)`: Evaluate on test set
- `save_model(filepath)`: Save trained model
- `load_model(filepath)`: Load trained model

### BrainCancerClassifier

```python
BrainCancerClassifier(
    input_shape=(224, 224, 3),
    num_classes=4,
    base_model_name="ResNet50",
    learning_rate=0.0001
)
```

Same methods as LungCancerClassifier.

### ImagePreprocessor

```python
ImagePreprocessor(
    target_size=(224, 224),
    normalize=True,
    clahe=False
)
```

**Methods:**
- `preprocess_image(image)`: Preprocess single image
- `load_and_preprocess_image(path)`: Load and preprocess from file
- `preprocess_batch(images)`: Preprocess batch of images
- `remove_noise(image, method)`: Remove noise from image
- `enhance_contrast(image, method)`: Enhance image contrast

### DataLoader

```python
DataLoader(
    data_dir,
    image_size=(224, 224),
    batch_size=32,
    validation_split=0.2
)
```

**Methods:**
- `load_dataset(subset, augment, shuffle)`: Load dataset
- `load_train_validation_datasets()`: Load train and validation sets
- `get_class_labels()`: Get class label names
- `get_dataset_info()`: Get dataset statistics

## 📈 Results

### Expected Performance

| Model Type | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|------------|----------|-----------|--------|----------|---------|
| Lung Cancer (Binary) | 92-95% | 90-93% | 88-92% | 89-92% | 0.95-0.98 |
| Brain Tumor (4-class) | 88-92% | 85-90% | 86-91% | 85-90% | 0.93-0.96 |

*Performance varies based on dataset quality and size*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with TensorFlow and Keras
- Transfer learning using ImageNet pre-trained models
- Inspired by state-of-the-art medical imaging research

## 📧 Contact

For questions and support, please open an issue on GitHub.

## ⚠️ Disclaimer

This system is designed for research and educational purposes. It should not be used as a sole diagnostic tool for medical decision-making. Always consult with qualified healthcare professionals for medical diagnosis and treatment.
