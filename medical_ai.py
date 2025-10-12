"""
Medical AI - Disease Detection from Medical Images
A comprehensive deep learning solution for medical image classification.
"""

import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class MedicalImageDataLoader:
    """Data loader for medical images with preprocessing and augmentation."""
    
    def __init__(self, image_size=(224, 224), batch_size=32):
        self.image_size = image_size
        self.batch_size = batch_size
    
    def load_data(self, data_path):
        """Load medical images from directory"""
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        train_generator = datagen.flow_from_directory(
            data_path,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        validation_generator = datagen.flow_from_directory(
            data_path,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        return train_generator, validation_generator
    
    def preprocess_single_image(self, image_path):
        """Preprocess a single image for prediction"""
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=self.image_size
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array


class MedicalAIModel:
    """Medical image classification model using CNN architecture."""
    
    def __init__(self, num_classes, input_shape=(224, 224, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
    
    def build_model(self):
        """Build CNN model for disease detection"""
        model = models.Sequential([
            # Convolutional Block 1
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=self.input_shape, padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Convolutional Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Convolutional Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        print("Model architecture created successfully!")
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with optimizer and loss"""
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), 
                    tf.keras.metrics.Precision(), 
                    tf.keras.metrics.Recall()]
        )
        print("Model compiled successfully!")
    
    def train(self, train_data, validation_data, epochs=50):
        """Train the model"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_medical_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return history
    
    def predict(self, image):
        """Make prediction on a single image"""
        prediction = self.model.predict(image)
        return prediction
    
    def save_model(self, filepath):
        """Save the model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a saved model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


class ModelEvaluator:
    """Evaluation and visualization tools for the medical AI model."""
    
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names
    
    def plot_training_history(self, history):
        """Plot training and validation metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Training')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Training')
        axes[0, 1].plot(history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # AUC
        if 'auc' in history.history:
            axes[1, 0].plot(history.history['auc'], label='Training')
            axes[1, 0].plot(history.history['val_auc'], label='Validation')
            axes[1, 0].set_title('Model AUC')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('AUC')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Precision
        if 'precision' in history.history:
            axes[1, 1].plot(history.history['precision'], label='Training')
            axes[1, 1].plot(history.history['val_precision'], label='Validation')
            axes[1, 1].set_title('Model Precision')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Precision')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
        print("Training history plot saved to training_history.png")
    
    def evaluate_model(self, test_data):
        """Evaluate model on test data"""
        y_pred = []
        y_true = []
        
        for images, labels in test_data:
            predictions = self.model.predict(images)
            y_pred.extend(np.argmax(predictions, axis=1))
            y_true.extend(np.argmax(labels, axis=1))
        
        return y_true, y_pred
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix - Disease Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()
        print("Confusion matrix saved to confusion_matrix.png")
    
    def generate_classification_report(self, y_true, y_pred):
        """Generate classification report"""
        report = classification_report(y_true, y_pred, 
                                      target_names=self.class_names)
        print("\nClassification Report:")
        print("=" * 60)
        print(report)
        return report


def main():
    """Example usage of the Medical AI system."""
    print("=" * 60)
    print("Medical AI - Disease Detection System")
    print("=" * 60)
    
    # Example configuration
    DATA_PATH = 'path/to/dataset'  # Replace with your dataset path
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    print("\nNote: Update DATA_PATH variable with your medical image dataset path")
    print("Expected directory structure:")
    print("  dataset/")
    print("    class1/")
    print("      image1.jpg")
    print("    class2/")
    print("      image2.jpg")
    print("=" * 60)


if __name__ == "__main__":
    main()
