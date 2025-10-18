"""Data loading utilities."""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array # Import specific Keras funcs

class DataLoader:
    """Utilities for loading medical image datasets."""
    
    # ... (init, load_dataset, load_train_validation_datasets, load_test_dataset methods are the same)

    def __init__(
        self,
        data_dir: str,
        image_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        validation_split: float = 0.2,
    ):
        """
        Initialize data loader.
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.validation_split = validation_split

    def load_dataset(
        self,
        subset: str = 'training',
        augment: bool = False,
        shuffle: bool = True
    ) -> tf.keras.preprocessing.image.DirectoryIterator:
        # ... (load_dataset implementation is the same)
        if augment:
            datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                shear_range=0.2,
                fill_mode='nearest',
                validation_split=self.validation_split
            )
        else:
            datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=self.validation_split
            )
        
        generator = datagen.flow_from_directory(
            self.data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='binary' if self._count_classes() == 2 else 'categorical',
            subset=subset,
            shuffle=shuffle
        )
        
        return generator

    def load_train_validation_datasets(
        self,
        augment_train: bool = True
    ) -> Tuple[tf.keras.preprocessing.image.DirectoryIterator, 
               tf.keras.preprocessing.image.DirectoryIterator]:
        # ... (load_train_validation_datasets implementation is the same)
        train_gen = self.load_dataset(
            subset='training',
            augment=augment_train,
            shuffle=True
        )
        
        val_gen = self.load_dataset(
            subset='validation',
            augment=False,
            shuffle=False
        )
        
        return train_gen, val_gen

    def load_test_dataset(
        self,
        test_dir: str,
        shuffle: bool = False
    ) -> tf.keras.preprocessing.image.DirectoryIterator:
        # ... (load_test_dataset implementation is the same)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        test_gen = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='binary' if self._count_classes() == 2 else 'categorical',
            shuffle=shuffle
        )
        
        return test_gen

    def load_images_from_directory(
        self,
        directory: str,
        preprocess_func: Optional[callable] = None
    ) -> Tuple[np.ndarray, List[str]]:
        # ... (load_images_from_directory implementation is the same)
        directory = Path(directory)
        images = []
        filenames = []
        
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        
        for img_path in directory.glob('**/*'):
            if img_path.suffix.lower() in valid_extensions:
                try:
                    img = load_img(
                        img_path,
                        target_size=self.image_size
                    )
                    img_array = img_to_array(img)
                    
                    if preprocess_func:
                        img_array = preprocess_func(img_array)
                    else:
                        img_array = img_array / 255.0
                    
                    images.append(img_array)
                    filenames.append(str(img_path))
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        return np.array(images), filenames

    # --- TASK 3: DEBUGGING UTILITY ---
    def get_debug_slice(
        self, 
        num_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load a small, deterministic slice of images directly into memory for debugging.

        Args:
            num_samples: Total number of samples to load across all classes.
        
        Returns:
            Tuple of (images_array, labels_array, class_labels)
        """
        all_images = []
        all_labels = []
        class_names = self.get_class_labels()
        num_classes = len(class_names)
        
        # Determine max samples per class (distributes load evenly)
        max_samples_per_class = max(1, num_samples // num_classes)
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = self.data_dir / class_name
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
            
            # Use glob to find all images, then slice to max_samples_per_class
            image_paths = [
                f for ext in valid_extensions 
                for f in class_dir.glob(f"*{ext}")
            ][:max_samples_per_class]

            for img_path in image_paths:
                try:
                    img = load_img(img_path, target_size=self.image_size)
                    img_array = img_to_array(img) / 255.0 # Simple normalization
                    
                    all_images.append(img_array)
                    all_labels.append(class_idx)
                except Exception as e:
                    print(f"Warning: Skipping {img_path} for debug slice: {e}")

        # Convert labels to one-hot if more than 2 classes
        labels_array = np.array(all_labels)
        if num_classes > 2:
            labels_array = tf.keras.utils.to_categorical(labels_array, num_classes=num_classes)
        
        print(f"Successfully loaded {len(all_images)} images for debug slice.")
        return np.array(all_images), labels_array, class_names

    def get_class_labels(self) -> List[str]:
        """
        Get class labels from directory structure.
        """
        classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        return classes
    
    def _count_classes(self) -> int:
        """Count number of classes in data directory."""
        return len([d for d in self.data_dir.iterdir() if d.is_dir()])
    
    def get_dataset_info(self) -> dict:
        """
        Get information about the dataset.
        """
        classes = self.get_class_labels()
        
        info = {
            "data_directory": str(self.data_dir),
            "num_classes": len(classes),
            "classes": classes,
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "validation_split": self.validation_split,
        }
        
        # Count images per class
        for class_name in classes:
            class_dir = self.data_dir / class_name
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
            num_images = sum(1 for f in class_dir.glob('**/*') 
                           if f.suffix.lower() in valid_extensions)
            info[f"{class_name}_count"] = num_images
        
        return info