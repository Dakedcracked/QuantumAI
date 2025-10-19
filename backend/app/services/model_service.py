from typing import Tuple
import numpy as np
from pathlib import Path

class ModelService:
    """Singleton-like service to load and run ML models."""

    def __init__(self, brain_model_path: str, lung_model_path: str):
        self.brain_model_path = brain_model_path
        self.lung_model_path = lung_model_path
        self._brain = None
        self._lung = None

    def _load_brain(self):
        if self._brain is None:
            # Lazy imports to avoid heavy deps at startup
            from src.models import EffResNetViTClassifier
            from src.utils.preprocessing import ImagePreprocessor
            from src.config.model_config import ModelConfig

            cfg = ModelConfig.brain_hybrid_default()
            model = EffResNetViTClassifier(
                input_shape=tuple(cfg.get("input_shape")),
                num_classes=cfg.get("num_classes"),
                base_model_name=cfg.get("base_model"),
                learning_rate=cfg.get("learning_rate"),
            )
            model.load_model(self.brain_model_path)
            model.set_class_labels(cfg.get("class_labels"))
            self._brain = (model, ImagePreprocessor(target_size=tuple(cfg.get("input_shape")[:2]), normalize=True))
        return self._brain

    def _load_lung(self):
        if self._lung is None:
            from src.models import EffResNetViTClassifier
            from src.utils.preprocessing import ImagePreprocessor
            from src.config.model_config import ModelConfig

            cfg = ModelConfig.lung_hybrid_default()
            model = EffResNetViTClassifier(
                input_shape=tuple(cfg.get("input_shape")),
                num_classes=cfg.get("num_classes"),
                base_model_name=cfg.get("base_model"),
                learning_rate=cfg.get("learning_rate"),
            )
            model.load_model(self.lung_model_path)
            model.set_class_labels(cfg.get("class_labels"))
            self._lung = (model, ImagePreprocessor(target_size=tuple(cfg.get("input_shape")[:2]), normalize=True))
        return self._lung

    def predict(self, model_type: str, image_path: str) -> Tuple[str, float, np.ndarray]:
        if model_type == "brain":
            model, pre = self._load_brain()
        elif model_type == "lung":
            model, pre = self._load_lung()
        else:
            raise ValueError("Invalid model_type. Use 'brain' or 'lung'.")

        img = pre.load_and_preprocess_image(image_path)
        preds, labels = model.predict_with_labels(np.array([img]))
        label = labels[0]
        conf = float(np.max(preds[0]))
        return label, conf, preds[0]
