"""Medical image classification models."""

from src.models.base_classifier import BaseClassifier
from src.models.effresnet_vit_classifier import EffResNetViTClassifier

__all__ = [
    "BaseClassifier",
    "EffResNetViTClassifier",
]