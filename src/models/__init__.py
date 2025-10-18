"""Medical image classification models."""

# REMOVE: from src.models.lung_cancer_classifier import LungCancerClassifier
# REMOVE: from src.models.brain_cancer_classifier import BrainCancerClassifier

from src.models.base_classifier import BaseClassifier
from src.models.effresnet_vit_classifier import EffResNetViTClassifier

__all__ = [
    # REMOVE: "LungCancerClassifier",
    # REMOVE: "BrainCancerClassifier",
    "BaseClassifier",
    "EffResNetViTClassifier",
]