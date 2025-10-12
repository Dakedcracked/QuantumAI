# QuantumAI Quick Start Guide

Get started with QuantumAI in just a few minutes!

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Dakedcracked/QuantumAI.git
cd QuantumAI

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Simple 3-Step Workflow

### Step 1: Prepare Your Data

Organize your medical images in the following structure:

```
data/lung_cancer/train/
├── Normal/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Cancerous/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### Step 2: Train Your Model

Run the training script:

```bash
python examples/train_lung_cancer_model.py
```

Or use Python directly:

```python
from src.models import LungCancerClassifier
from src.data import DataLoader

# Load data
loader = DataLoader("data/lung_cancer/train", image_size=(224, 224))
train_data, val_data = loader.load_train_validation_datasets()

# Create and train model
model = LungCancerClassifier()
model.build_model()
model.train(train_data, val_data, epochs=50)

# Save trained model
model.save_model("models/saved/my_model.h5")
```

### Step 3: Make Predictions

```python
from src.models import LungCancerClassifier
from src.utils import ImagePreprocessor
import numpy as np

# Load trained model
model = LungCancerClassifier()
model.load_model("models/saved/my_model.h5")

# Load and preprocess image
preprocessor = ImagePreprocessor(target_size=(224, 224))
image = preprocessor.load_and_preprocess_image("path/to/scan.jpg")

# Predict
predictions, labels = model.predict_with_labels(np.array([image]))
print(f"Prediction: {labels[0]}")
print(f"Confidence: {predictions[0][0]:.2%}")
```

## 🧠 Brain Tumor Detection

For brain tumor classification, use the brain cancer classifier:

```python
from src.models import BrainCancerClassifier

# Initialize with 4 classes (Glioma, Meningioma, No Tumor, Pituitary)
model = BrainCancerClassifier(num_classes=4)
model.build_model()

# Rest of the workflow is the same!
```

## 🎨 Visualization

Visualize your results easily:

```python
from src.utils import Visualizer

visualizer = Visualizer()

# Plot training history
visualizer.plot_training_history(
    history,
    metrics=['loss', 'accuracy'],
    save_path="results/training.png"
)

# Plot predictions
visualizer.plot_images(
    images=test_images,
    predictions=predicted_labels,
    true_labels=true_labels,
    cols=4
)
```

## 📊 Model Evaluation

Evaluate your model's performance:

```python
from src.utils import ModelEvaluator

evaluator = ModelEvaluator(class_labels=["Normal", "Cancerous"])

# Get comprehensive metrics
metrics = evaluator.evaluate_binary_classification(y_true, y_pred)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1-Score: {metrics['f1_score']:.4f}")
```

## ⚙️ Configuration

Use YAML configs for easy customization:

```python
from src.config import ModelConfig

# Load config
config = ModelConfig.from_yaml("configs/lung_cancer_config.yaml")

# Or create custom config
config = ModelConfig({
    "base_model": "DenseNet121",
    "learning_rate": 0.00005,
    "epochs": 100,
    "batch_size": 16
})

# Save config
config.save_yaml("configs/my_config.yaml")
```

## 🔧 Advanced Features

### Data Augmentation

```python
from src.data import DataAugmentor

augmentor = DataAugmentor()
augmented_image = augmentor.augment(
    image,
    apply_rotation=True,
    apply_flip=True,
    apply_zoom=True
)
```

### Different Base Architectures

```python
# Try different pre-trained models
model = LungCancerClassifier(base_model_name="DenseNet121")
model = LungCancerClassifier(base_model_name="EfficientNetB0")
model = LungCancerClassifier(base_model_name="InceptionV3")
```

### Transfer Learning & Fine-Tuning

```python
# Train with frozen base
model.build_model(freeze_base=True)
model.train(train_data, val_data, epochs=30)

# Fine-tune entire model
model.model.trainable = True
model.model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.train(train_data, val_data, epochs=20)
```

## 📁 Example Scripts

We provide ready-to-use examples:

```bash
# Train models
python examples/train_lung_cancer_model.py
python examples/train_brain_cancer_model.py

# Make predictions
python examples/predict_lung_cancer.py
python examples/predict_brain_cancer.py

# System demonstration
python examples/demo_system.py
```

## 🎓 Recommended Datasets

### Lung Cancer
- **Kaggle**: IQ-OTHNCCD Lung Cancer Dataset
- **LIDC-IDRI**: Lung Image Database Consortium
- **Cancer Imaging Archive**: Various lung cancer datasets

### Brain Tumor
- **Kaggle**: Brain Tumor MRI Dataset
- **BraTS**: Brain Tumor Segmentation Challenge
- **TCIA**: The Cancer Imaging Archive

## 💡 Tips for Best Results

1. **Data Quality**: Use high-resolution medical scans
2. **Balance**: Ensure similar numbers of images per class
3. **Augmentation**: Enable augmentation to prevent overfitting
4. **Epochs**: Start with 50 epochs, adjust based on validation loss
5. **Learning Rate**: Use 0.0001 for initial training
6. **Fine-tuning**: After initial training, unfreeze and train with lower LR

## 🐛 Common Issues

### Out of Memory
```python
# Reduce batch size
loader = DataLoader(data_dir="...", batch_size=8)
```

### Overfitting
```python
# Use data augmentation and dropout
train_data = loader.load_dataset(augment=True)
model.build_model()  # Has built-in dropout
```

### Poor Performance
- Check data quality and labels
- Increase training epochs
- Try different base architectures
- Ensure proper data augmentation

## 📚 Learn More

- **Full Documentation**: See [README.md](README.md)
- **Detailed Guide**: Check [docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md)
- **Implementation Details**: Read [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

## ⚠️ Important Note

This system is for **research and educational purposes only**. Do not use as the sole diagnostic tool for medical decisions. Always consult qualified healthcare professionals.

## 🤝 Need Help?

- Check the documentation
- Look at example scripts
- Open an issue on GitHub

---

**Ready to start?** Run `python examples/demo_system.py` to see the system in action!
