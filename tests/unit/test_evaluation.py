"""
Tests for model evaluation functionality in src/models/evaluation.py
"""
import pytest
import numpy as np
import tempfile
import os
from src.models.evaluation import (
    evaluate_predictions,
    get_classification_report,
    create_confusion_matrix
)


class TestEvaluatePredictions:
    """Tests for evaluate_predictions function"""

    def test_evaluate_predictions_returns_all_metrics(self):
        """Test that evaluate_predictions returns all required metrics"""
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 1, 0, 1])
        
        # Use the evaluate_predictions function
        metrics = evaluate_predictions(y_true, y_pred)
        
        # All metrics should be calculated successfully
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert 'recall' in metrics
        
        assert metrics['accuracy'] is not None
        assert metrics['f1_score'] is not None
        assert metrics['recall'] is not None
        
        # Metrics should be in valid range
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        assert 0 <= metrics['recall'] <= 1

    def test_evaluate_predictions_perfect_score(self):
        """Test evaluate_predictions with perfect predictions"""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])
        
        metrics = evaluate_predictions(y_true, y_pred)
        
        # All metrics should be 1.0 for perfect predictions
        assert metrics['accuracy'] == 1.0
        assert metrics['f1_score'] == 1.0
        assert metrics['recall'] == 1.0
    
    def test_evaluate_predictions_with_typical_data(self):
        """Test evaluate_predictions with typical prediction scenario"""
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 1])
        
        # Calculate metrics using the function
        metrics = evaluate_predictions(y_true, y_pred)
        
        # Check that all metrics are present
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert 'recall' in metrics
        
        # Check that all metrics are in valid range
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        assert 0 <= metrics['recall'] <= 1


class TestClassificationReport:
    """Tests for get_classification_report function"""
    
    def test_classification_report_contains_expected_info(self):
        """Test that get_classification_report contains expected information"""
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 1, 0, 1])
        
        report = get_classification_report(y_true, y_pred)
        
        # Report should contain key metrics
        assert 'precision' in report
        assert 'recall' in report
        assert 'f1-score' in report
        assert 'support' in report

    def test_classification_report_is_string(self):
        """Test that get_classification_report returns a string"""
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 1])
        
        # Get classification report using the function
        report = get_classification_report(y_true, y_pred)
        
        # Check that report is a string and contains expected content
        assert isinstance(report, str)
        assert len(report) > 0


class TestConfusionMatrix:
    """Tests for create_confusion_matrix function"""

    def test_create_confusion_matrix_file_created(self):
        """Test that create_confusion_matrix creates a file"""
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 1])
        
        # Create temporary file for confusion matrix
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            tmp_path = tmp.name
        
        try:
            # Create confusion matrix using the function
            create_confusion_matrix(y_true, y_pred, tmp_path)
            
            # Check that file was created
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 0  # File should have content
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_create_confusion_matrix_with_custom_labels(self):
        """Test create_confusion_matrix with custom labels"""
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 1])
        
        # Create temporary file for confusion matrix
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            tmp_path = tmp.name
        
        try:
            # Create confusion matrix with custom labels
            create_confusion_matrix(y_true, y_pred, tmp_path, labels=['Negative', 'Positive'])
            
            # Check that file was created
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 0
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

