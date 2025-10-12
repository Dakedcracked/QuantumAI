"""Visualization utilities for medical imaging."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple
import cv2


class Visualizer:
    """Utilities for visualizing medical images and results."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_images(
        self,
        images: List[np.ndarray],
        titles: Optional[List[str]] = None,
        predictions: Optional[List[str]] = None,
        true_labels: Optional[List[str]] = None,
        cols: int = 4,
        save_path: Optional[str] = None
    ):
        """
        Plot multiple images in a grid.
        
        Args:
            images: List of images to plot
            titles: Optional list of titles
            predictions: Optional list of predictions
            true_labels: Optional list of true labels
            cols: Number of columns in grid
            save_path: Path to save the figure
        """
        n_images = len(images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = axes.flatten() if n_images > 1 else [axes]
        
        for idx, ax in enumerate(axes):
            if idx < n_images:
                img = images[idx]
                
                # Ensure proper format for display
                if img.dtype == np.float32 or img.dtype == np.float64:
                    img = (img * 255).astype(np.uint8)
                
                ax.imshow(img)
                ax.axis('off')
                
                # Set title
                title = ""
                if titles and idx < len(titles):
                    title = titles[idx]
                elif predictions and true_labels and idx < len(predictions):
                    pred = predictions[idx]
                    true = true_labels[idx]
                    color = 'green' if pred == true else 'red'
                    title = f"Pred: {pred}\nTrue: {true}"
                    ax.set_title(title, color=color, fontsize=10)
                elif predictions and idx < len(predictions):
                    title = f"Prediction: {predictions[idx]}"
                
                if title and not (predictions and true_labels):
                    ax.set_title(title, fontsize=10)
            else:
                ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_labels: List[str],
        normalize: bool = False,
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            class_labels: List of class labels
            normalize: Whether to normalize the matrix
            save_path: Path to save the figure
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
        else:
            fmt = 'd'
        
        plt.figure(figsize=self.figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_training_history(
        self,
        history: dict,
        metrics: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot training history.
        
        Args:
            history: Training history dictionary
            metrics: List of metrics to plot
            save_path: Path to save the figure
        """
        if metrics is None:
            metrics = [key for key in history.keys() if not key.startswith('val_')]
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Plot training metric
            ax.plot(history[metric], label=f'Training {metric}', linewidth=2)
            
            # Plot validation metric if available
            val_metric = f'val_{metric}'
            if val_metric in history:
                ax.plot(history[val_metric], label=f'Validation {metric}', linewidth=2)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} over Epochs')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_roc_curve(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc_score: float,
        title: str = "ROC Curve",
        save_path: Optional[str] = None
    ):
        """
        Plot ROC curve.
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            auc_score: AUC score
            title: Plot title
            save_path: Path to save the figure
        """
        plt.figure(figsize=self.figsize)
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_prediction_distribution(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        class_labels: List[str],
        save_path: Optional[str] = None
    ):
        """
        Plot distribution of predictions.
        
        Args:
            predictions: Predicted class indices
            true_labels: True class indices
            class_labels: List of class labels
            save_path: Path to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Plot predictions
        unique, counts = np.unique(predictions, return_counts=True)
        ax1.bar(unique, counts, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.set_title('Predicted Distribution')
        ax1.set_xticks(range(len(class_labels)))
        ax1.set_xticklabels(class_labels, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Plot true labels
        unique, counts = np.unique(true_labels, return_counts=True)
        ax2.bar(unique, counts, color='coral', alpha=0.7)
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Count')
        ax2.set_title('True Distribution')
        ax2.set_xticks(range(len(class_labels)))
        ax2.set_xticklabels(class_labels, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_comparison(
        self,
        results: dict,
        metric: str = 'accuracy',
        save_path: Optional[str] = None
    ):
        """
        Plot comparison of multiple models.
        
        Args:
            results: Dictionary of model names to metrics
            metric: Metric to compare
            save_path: Path to save the figure
        """
        models = list(results.keys())
        scores = [results[model][metric] for model in models]
        
        plt.figure(figsize=self.figsize)
        bars = plt.bar(models, scores, color='steelblue', alpha=0.7)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('Model')
        plt.ylabel(metric.capitalize())
        plt.title(f'Model Comparison - {metric.capitalize()}')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
