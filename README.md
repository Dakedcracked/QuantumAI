# QuantumAI
This is the main repository of the Quantum AI, which is highly focused on image-based recognition of diseases and predictive modeling. It analyzes scanned medical images to detect and diagnose diseases and predict future health conditions.

The purpose of Quantum AI is to make research and disease prediction simple for clinics and large-scaled medical hospitals.

## Features

- **Medical Image Classification**: Deep learning CNN model for disease detection from medical images
- **Data Preprocessing**: Automated image loading and preprocessing with data augmentation
- **Model Training**: Comprehensive training pipeline with callbacks and optimization
- **Model Evaluation**: Performance metrics, confusion matrix, and classification reports
- **Prediction**: Single image prediction with confidence scores

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Dakedcracked/QuantumAI.git
cd QuantumAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Dataset Structure

Organize your medical images in the following structure:
```
dataset/
  disease1/
    image1.jpg
    image2.jpg
  disease2/
    image1.jpg
    image2.jpg
  normal/
    image1.jpg
    image2.jpg
```

### Training the Model

Open `model.ipynb` in Jupyter Notebook and follow these steps:

1. **Load Data**:
```python
data_loader = MedicalImageDataLoader(image_size=(224, 224), batch_size=32)
train_data, val_data = data_loader.load_data('path/to/dataset')
```

2. **Build and Compile Model**:
```python
num_classes = len(train_data.class_indices)
medical_ai = MedicalAIModel(num_classes=num_classes)
medical_ai.build_model()
medical_ai.compile_model(learning_rate=0.001)
```

3. **Train Model**:
```python
history = medical_ai.train(train_data, val_data, epochs=50)
```

4. **Evaluate Model**:
```python
class_names = list(train_data.class_indices.keys())
evaluator = ModelEvaluator(medical_ai.model, class_names)
evaluator.plot_training_history(history)

y_true, y_pred = evaluator.evaluate_model(val_data)
evaluator.plot_confusion_matrix(y_true, y_pred)
evaluator.generate_classification_report(y_true, y_pred)
```

5. **Make Predictions**:
```python
image = data_loader.preprocess_single_image('path/to/test/image.jpg')
prediction = medical_ai.predict(image)
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction) * 100
print(f'Predicted Disease: {predicted_class}')
print(f'Confidence: {confidence:.2f}%')
```

### Saving and Loading Models

Save trained model:
```python
medical_ai.save_model('medical_ai_model.h5')
```

Load saved model:
```python
medical_ai.load_model('medical_ai_model.h5')
```

## Model Architecture

The CNN architecture consists of:
- 3 Convolutional blocks with batch normalization and dropout
- MaxPooling layers for dimensionality reduction
- Dense layers with regularization
- Softmax activation for multi-class classification

## Dependencies

- TensorFlow >= 2.12.0
- NumPy >= 1.23.0
- scikit-learn >= 1.2.0
- Matplotlib >= 3.7.0
- Seaborn >= 0.12.0
- Pandas >= 2.0.0
- Pillow >= 9.5.0

## License

This project is open source and available for research and educational purposes.