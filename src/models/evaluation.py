"""
Model evaluation utilities for calculating metrics, generating reports, and visualizations.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score, confusion_matrix, ConfusionMatrixDisplay


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate predictions and calculate metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with accuracy, f1_score, and recall metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred)
    }
    
    return metrics


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    Generate a classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Classification report as string
    """
    return classification_report(y_true, y_pred)


def create_confusion_matrix(y_true: np.ndarray, 
                           y_pred: np.ndarray, 
                           output_path: str,
                           labels: Optional[list] = None,
                           title: Optional[str] = None) -> str:
    """
    Create and save a confusion matrix plot.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Path where to save the confusion matrix image
        labels: Display labels for the matrix (default: ['Low', 'High'])
        title: Optional title for the plot
        
    Returns:
        Path to the saved plot
    """
    if labels is None:
        labels = ['Low', 'High']
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues')
    
    if title:
        plt.title(title)
        plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()
    
    return output_path

