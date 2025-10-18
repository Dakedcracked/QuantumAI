"""EffResNet-ViT hybrid classifier.

Combines an EfficientNet backbone (optionally fused with ResNet) and a small
Vision-Transformer encoder on top of spatial features, then a classification head.
"""

from typing import Dict, Optional, Tuple, List
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0, ResNet50

from src.models.base_classifier import BaseClassifier


class EffResNetViTClassifier(BaseClassifier):
    """Hybrid EfficientNet/ResNet + ViT classifier.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 2,
        base_model_name: str = "EfficientNetB0",
        learning_rate: float = 1e-4,
        embed_dim: int = 192,
        num_transformer_layers: int = 4,
        num_heads: int = 4,
        mlp_dim: int = 384,
        dropout: float = 0.1,
    ):
        super().__init__(
            input_shape=input_shape,
            num_classes=num_classes,
            base_model_name=base_model_name,
            learning_rate=learning_rate,
        )

        self.embed_dim = embed_dim
        self.num_transformer_layers = num_transformer_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        # Initialize class_labels as None or empty list; they must be set from config
        self.class_labels: List[str] = []

    def set_class_labels(self, labels: List[str]): # <--- NEW METHOD
        """Set human-readable class labels from the configuration."""
        if len(labels) != self.num_classes:
             raise ValueError(f"Label count ({len(labels)}) does not match num_classes ({self.num_classes})")
        self.class_labels = labels

    def get_model_info(self) -> Dict[str, str]:
        return {
            "model_type": "EffResNet-ViT Hybrid Classifier",
            "base_architecture": self.base_model_name,
            "input_shape": str(self.input_shape),
            "num_classes": str(self.num_classes),
            "embed_dim": str(self.embed_dim),
            "transformer_layers": str(self.num_transformer_layers),
            "num_heads": str(self.num_heads),
            "description": "EfficientNet/ResNet fused backbone with a small ViT encoder for medical image classification",
            "class_labels": ", ".join(self.class_labels) # Include labels in info
        }

    # [Rest of the methods: _build_vit_encoder, build_model, predict_with_labels]
    def _build_vit_encoder(self, x, patch_size: int = 1):
        """Convert spatial features to sequence and apply transformer encoder."""
        # Convert spatial features to a sequence dynamically: (B, H*W, C)
        seq = layers.Lambda(lambda t: tf.reshape(t, (tf.shape(t)[0], -1, tf.shape(t)[-1])))(x)

        # Linear projection to embed_dim
        seq = layers.Dense(self.embed_dim)(seq)

        # Add sinusoidal positional encoding (works with dynamic seq length)
        def add_sinusoidal_pos_encoding(tensor):
            # tensor shape: (B, S, D)
            seq_len = tf.shape(tensor)[1]
            d_model = self.embed_dim
            positions = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32)  # (S,1)
            dims = tf.cast(tf.range(d_model)[tf.newaxis, :], tf.float32)  # (1,D)
            angle_rates = 1 / tf.pow(10000.0, (2 * (dims // 2)) / tf.cast(d_model, tf.float32))  # (1,D)
            angle_rads = positions * angle_rates  # (S,D)

            # apply sin to even indices; cos to odd indices
            sines = tf.sin(angle_rads[:, 0::2])
            cosines = tf.cos(angle_rads[:, 1::2])
            # interleave sines and cosines to form (S, D)
            pos_encoding = tf.reshape(tf.concat([sines, cosines], axis=-1), (seq_len, d_model))
            pos_encoding = pos_encoding[tf.newaxis, :, :]  # (1, S, D)
            return tensor + tf.cast(pos_encoding, tensor.dtype)

        seq = layers.Lambda(add_sinusoidal_pos_encoding)(seq)

        # Transformer encoder blocks
        for _ in range(self.num_transformer_layers):
            # Multi-head self-attention
            x1 = layers.LayerNormalization(epsilon=1e-6)(seq)
            x1 = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim // self.num_heads)(x1, x1)
            x1 = layers.Dropout(self.dropout)(x1)
            seq = layers.Add()([seq, x1])

            # MLP
            x2 = layers.LayerNormalization(epsilon=1e-6)(seq)
            x2 = layers.Dense(self.mlp_dim, activation="gelu")(x2)
            x2 = layers.Dropout(self.dropout)(x2)
            x2 = layers.Dense(self.embed_dim)(x2)
            x2 = layers.Dropout(self.dropout)(x2)
            seq = layers.Add()([seq, x2])

        # Pool sequence: use mean pooling
        seq = layers.LayerNormalization(epsilon=1e-6)(seq)
        pooled = layers.GlobalAveragePooling1D()(seq)
        return pooled

    def build_model(self, freeze_base: bool = True, weights: Optional[str] = "imagenet") -> Model:
        """Build the hybrid model and compile it.
        """
        inputs = keras.Input(shape=self.input_shape)

        # Primary backbone: EfficientNetB0
        eff = EfficientNetB0(weights=weights, include_top=False, input_shape=self.input_shape)
        eff.trainable = not freeze_base
        eff_feat = eff(inputs)

        # Optional ResNet fusion for 'EffResNet' behavior
        if self.base_model_name and self.base_model_name.lower().startswith("effresnet"):
            res = ResNet50(weights=weights, include_top=False, input_shape=self.input_shape)
            res.trainable = not freeze_base
            res_feat = res(inputs)
            # Resize features if necessary by a 1x1 conv to match channels
            # Concatenate along channels
            # Ensure same spatial dims: assume both produce compatible feature map sizes
            fused = layers.Concatenate(axis=-1)([eff_feat, res_feat])
            x = fused
        else:
            x = eff_feat

        # Optional conv to reduce channels before transformer
        x = layers.Conv2D(self.embed_dim, kernel_size=1, activation="relu", name="last_cnn_features")(x) # <--- NAMED FOR Grad-CAM

        # Apply ViT-like encoder
        pooled = self._build_vit_encoder(x)

        # Classification head
        x = layers.Dense(512, activation="relu")(pooled)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.2)(x)

        if self.num_classes == 2:
            outputs = layers.Dense(1, activation="sigmoid")(x)
            loss = "binary_crossentropy"
        else:
            outputs = layers.Dense(self.num_classes, activation="softmax")(x)
            loss = "categorical_crossentropy"

        model = Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=loss,
            metrics=["accuracy", keras.metrics.AUC(name="auc"), keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")],
        )

        self.model = model
        return model

    def predict_with_labels(self, images):
        preds = self.predict(images)
        if not self.class_labels:
             raise ValueError("Class labels must be set using set_class_labels() before prediction.")
             
        if self.num_classes == 2:
            labels_idx = (preds > 0.5).astype(int).flatten()
        else:
            labels_idx = preds.argmax(axis=1)
        labels = [self.class_labels[i] for i in labels_idx]
        return preds, labels