"""Data loading utilities."""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataLoader:
    """Utilities for loading medical image datasets."""
    
    def __init__(
        self,
        data_dir: str,
        image_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        validation_split: float = 0.2,
    ):
        """
        Initialize data loader.
        
        Args:
            data_dir: Root directory containing image data
            image_size: Target image size (width, height)
            batch_size: Batch size for data generators
            validation_split: Fraction of data to use for validation
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
        """
        Load dataset from directory structure.
        
        Expected structure:
        data_dir/
            class1/
                image1.jpg
                image2.jpg
            class2/
                image1.jpg
                image2.jpg
        
        Args:
            subset: 'training' or 'validation'
            augment: Whether to apply data augmentation
            shuffle: Whether to shuffle data
            
        Returns:
            Data generator
        """
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
        """
        Load both training and validation datasets.
        
        Args:
            augment_train: Whether to augment training data
            
        Returns:
            Tuple of (train_generator, validation_generator)
        """
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
        """
        Load test dataset.
        
        Args:
            test_dir: Directory containing test images
            shuffle: Whether to shuffle data
            
        Returns:
            Test data generator
        """
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
        """
        Load all images from a directory.
        
        Args:
            directory: Directory path
            preprocess_func: Optional preprocessing function
            
        Returns:
            Tuple of (images array, filenames list)
        """
        directory = Path(directory)
        images = []
        filenames = []
        
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        
        for img_path in directory.glob('**/*'):
            if img_path.suffix.lower() in valid_extensions:
                try:
                    img = tf.keras.preprocessing.image.load_img(
                        img_path,
                        target_size=self.image_size
                    )
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    
                    if preprocess_func:
                        img_array = preprocess_func(img_array)
                    else:
                        img_array = img_array / 255.0
                    
                    images.append(img_array)
                    filenames.append(str(img_path))
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        return np.array(images), filenames
    
    def get_class_labels(self) -> List[str]:
        """
        Get class labels from directory structure.
        
        Returns:
            List of class labels
        """
        classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        return classes
    
    def _count_classes(self) -> int:
        """Count number of classes in data directory."""
        return len([d for d in self.data_dir.iterdir() if d.is_dir()])
    
    def get_dataset_info(self) -> dict:
        """
        Get information about the dataset.
        
        Returns:
            Dictionary with dataset information
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
