# Data Directory

This directory contains medical imaging datasets for training and testing the QuantumAI classification models.

## Directory Structure

```
data/
├── lung_cancer/
│   ├── train/
│   │   ├── Normal/
│   │   └── Cancerous/
│   ├── test/
│   │   ├── Normal/
│   │   └── Cancerous/
│   └── validation/
│       ├── Normal/
│       └── Cancerous/
└── brain_cancer/
    ├── train/
    │   ├── Glioma/
    │   ├── Meningioma/
    │   ├── No_Tumor/
    │   └── Pituitary/
    ├── test/
    │   ├── Glioma/
    │   ├── Meningioma/
    │   ├── No_Tumor/
    │   └── Pituitary/
    └── validation/
        ├── Glioma/
        ├── Meningioma/
        ├── No_Tumor/
        └── Pituitary/
```

## Dataset Requirements

### Lung Cancer Dataset

- **Classes**: 2 (Normal, Cancerous) or more for multi-class
- **Image Type**: CT scans, X-rays
- **Format**: JPEG, PNG, BMP, TIFF
- **Recommended Size**: 224x224 pixels or higher
- **Recommended Samples**: Minimum 500 images per class

### Brain Cancer Dataset

- **Classes**: 4 (Glioma, Meningioma, No Tumor, Pituitary)
- **Image Type**: MRI scans (T1, T2, FLAIR)
- **Format**: JPEG, PNG, BMP, TIFF
- **Recommended Size**: 224x224 pixels or higher
- **Recommended Samples**: Minimum 500 images per class

## Public Datasets

You can use the following public datasets:

### Lung Cancer
- **IQ-OTHNCCD Lung Cancer Dataset**: Available on Kaggle
- **LIDC-IDRI**: Lung Image Database Consortium
- **Cancer Imaging Archive**: Multiple lung cancer datasets

### Brain Tumor
- **Brain Tumor MRI Dataset**: Available on Kaggle
- **BraTS (Brain Tumor Segmentation)**: Challenge dataset
- **TCIA (The Cancer Imaging Archive)**: Brain tumor collections

## Preparing Your Data

1. **Download Dataset**: Obtain medical imaging data from authorized sources
2. **Organize**: Place images in the appropriate class folders
3. **Anonymize**: Ensure patient data is removed or anonymized
4. **Quality Check**: Remove corrupted or low-quality images
5. **Balance**: Try to have similar numbers of images per class
6. **Split**: Divide into train (70-80%), validation (10-15%), test (10-15%)

## Data Privacy & Ethics

⚠️ **Important**: 
- Ensure you have proper authorization to use medical imaging data
- Remove all patient identifying information (HIPAA compliance)
- Follow institutional ethics guidelines
- Obtain proper consent if required
- Use datasets only for research/educational purposes

## Adding Your Data

1. Create the appropriate directory structure
2. Place your images in the correct class folders
3. Verify data organization:
   ```bash
   python -c "from src.data import DataLoader; loader = DataLoader('data/lung_cancer/train'); print(loader.get_dataset_info())"
   ```

## Notes

- Images will be automatically resized to the model's input size
- Both grayscale and RGB images are supported
- Data augmentation is applied automatically during training
- The system handles class imbalance automatically
