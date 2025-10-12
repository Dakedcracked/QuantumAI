# QuantumAI Tests

This directory contains test files for the QuantumAI system.

## Test Structure

```
tests/
├── test_models.py          # Model tests
├── test_data_loader.py     # Data loading tests
├── test_preprocessing.py   # Preprocessing tests
├── test_evaluation.py      # Evaluation tests
└── test_config.py          # Configuration tests
```

## Running Tests

### Install Test Dependencies

```bash
pip install pytest pytest-cov
```

### Run All Tests

```bash
# From project root
pytest tests/

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Run Specific Test Files

```bash
pytest tests/test_models.py
pytest tests/test_preprocessing.py
```

## Writing Tests

Tests should follow these guidelines:

1. **Use pytest**: All tests use pytest framework
2. **Descriptive names**: Test function names should describe what they test
3. **Arrange-Act-Assert**: Follow AAA pattern
4. **Mock external dependencies**: Use mocks for TensorFlow operations
5. **Test edge cases**: Include tests for error conditions

## Test Coverage Goals

- **Models**: Test initialization, building, configuration
- **Data Loading**: Test data loading, augmentation, preprocessing
- **Utilities**: Test all utility functions
- **Configuration**: Test config loading and saving
- **Integration**: Test end-to-end workflows

## Example Test

```python
import pytest
from src.models import LungCancerClassifier

def test_lung_cancer_classifier_initialization():
    """Test that LungCancerClassifier initializes correctly."""
    model = LungCancerClassifier(
        input_shape=(224, 224, 3),
        num_classes=2
    )
    assert model.num_classes == 2
    assert model.input_shape == (224, 224, 3)

def test_lung_cancer_classifier_class_labels():
    """Test that class labels are set correctly."""
    model = LungCancerClassifier(num_classes=2)
    assert len(model.class_labels) == 2
    assert "Normal" in model.class_labels
    assert "Cancerous" in model.class_labels
```

## Notes

- Tests require TensorFlow to be installed
- Some tests may require GPU for full functionality
- Data tests use small sample datasets
- Mock data is used where appropriate to speed up tests
