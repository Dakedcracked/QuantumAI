"""Model evaluation utilities."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)


class ModelEvaluator:
    """Utilities for evaluating classification models."""
    
    def __init__(self, class_labels: Optional[List[str]] = None):
        """
        Initialize the model evaluator.
        
        Args:
            class_labels: List of class label names
        """
        self.class_labels = class_labels
    
    def evaluate_binary_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Evaluate binary classification performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            threshold: Classification threshold
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Convert probabilities to binary predictions
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred_binary),
            "precision": precision_score(y_true, y_pred_binary, average='binary', zero_division=0),
            "recall": recall_score(y_true, y_pred_binary, average='binary', zero_division=0),
            "f1_score": f1_score(y_true, y_pred_binary, average='binary', zero_division=0),
            "specificity": self._calculate_specificity(y_true, y_pred_binary),
        }
        
        # Calculate AUC if predictions are probabilities
        if len(np.unique(y_pred)) > 2:
            metrics["auc_roc"] = roc_auc_score(y_true, y_pred)
        
        return metrics
    
    def evaluate_multiclass_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = 'weighted'
    ) -> Dict[str, float]:
        """
        Evaluate multi-class classification performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted class indices
            average: Averaging method for metrics
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
        }
        
        return metrics
    
    def get_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Generate confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)
    
    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_dict: bool = False
    ):
        """
        Generate detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            output_dict: Return as dictionary instead of string
            
        Returns:
            Classification report
        """
        return classification_report(
            y_true,
            y_pred,
            target_names=self.class_labels,
            output_dict=output_dict,
            zero_division=0
        )
    
    def calculate_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        return roc_curve(y_true, y_pred_proba)
    
    def calculate_sensitivity_specificity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate sensitivity (recall) and specificity.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Tuple of (sensitivity, specificity)
        """
        sensitivity = recall_score(y_true, y_pred, average='binary', zero_division=0)
        specificity = self._calculate_specificity(y_true, y_pred)
        
        return sensitivity, specificity
    
    def _calculate_specificity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calculate specificity (true negative rate).
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Specificity score
        """
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape[0] == 2:
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            # For multi-class, calculate mean specificity
            specificities = []
            for i in range(cm.shape[0]):
                tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
                fp = np.sum(cm[:, i]) - cm[i, i]
                specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
            specificity = np.mean(specificities)
        
        return specificity
    
    def compare_models(
        self,
        results: Dict[str, Dict[str, float]]
    ) -> Dict[str, str]:
        """
        Compare multiple models and identify best performers.
        
        Args:
            results: Dictionary of model names to their metric dictionaries
            
        Returns:
            Dictionary of metrics to best model names
        """
        best_models = {}
        
        # Get all metrics
        all_metrics = set()
        for metrics in results.values():
            all_metrics.update(metrics.keys())
        
        # Find best model for each metric
        for metric in all_metrics:
            best_score = -1
            best_model = None
            
            for model_name, metrics in results.items():
                if metric in metrics and metrics[metric] > best_score:
                    best_score = metrics[metric]
                    best_model = model_name
            
            best_models[metric] = best_model
        
        return best_models
