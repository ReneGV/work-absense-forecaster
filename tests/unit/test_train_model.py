"""
Unit tests for train_model module.

These tests demonstrate how the refactored code can be easily tested
with mocked dependencies and controlled test data.
"""

import pytest
import pandas as pd
import numpy as np
import joblib
from unittest.mock import Mock, patch, MagicMock
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import tempfile
import os

from src.models.train_model import (
    load_and_prepare_data,
    create_preprocessing_pipeline,
    get_default_models,
    train_and_evaluate_model,
    train_models,
    save_model
)
from src.models.evaluation import create_confusion_matrix


class TestLoadAndPrepareData:
    """Test data loading and preparation."""
    
    def test_load_and_prepare_data_returns_correct_shapes(self, sample_csv_file):
        """Test that data is split correctly."""
        X_train, X_test, y_train, y_test, median = load_and_prepare_data(
            sample_csv_file,
            test_size=0.2,
            random_state=42
        )
        
        # Check shapes
        assert len(X_train) + len(X_test) == 100  # Total samples
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
        
        # Check test size ratio
        assert len(X_test) == 20
        assert len(X_train) == 80
    
    def test_load_and_prepare_data_converts_to_binary(self, sample_csv_file):
        """Test that target is converted to binary based on median."""
        X_train, X_test, y_train, y_test, median = load_and_prepare_data(
            sample_csv_file,
            random_state=42
        )
        
        # Check that y contains only 0 and 1
        assert set(y_train.unique()).issubset({0, 1})
        assert set(y_test.unique()).issubset({0, 1})
        
        # Check median is returned
        assert isinstance(median, (int, float))
        assert median > 0
    
    def test_load_and_prepare_data_normalizes_column_names(self, sample_csv_file):
        """Test that column names are normalized."""
        X_train, X_test, y_train, y_test, median = load_and_prepare_data(
            sample_csv_file,
            random_state=42
        )
        
        # Check that column names are lowercase with underscores
        for col in X_train.columns:
            assert col == col.lower()
            assert ' ' not in col
    
    def test_load_and_prepare_data_with_different_test_size(self, sample_csv_file):
        """Test with different test size."""
        X_train, X_test, y_train, y_test, median = load_and_prepare_data(
            sample_csv_file,
            test_size=0.3,
            random_state=42
        )
        
        assert len(X_test) == 30
        assert len(X_train) == 70


class TestCreatePreprocessingPipeline:
    """Test preprocessing pipeline creation."""
    
    def test_create_preprocessing_pipeline_returns_pipeline(self):
        """Test that function returns a Pipeline object."""
        columns_to_drop = ['id']
        numerical_columns = ['age', 'weight']
        categorical_columns = ['social_drinker']
        
        pipeline = create_preprocessing_pipeline(
            columns_to_drop,
            numerical_columns,
            categorical_columns
        )
        
        assert isinstance(pipeline, Pipeline)
        assert 'drop_columns' in pipeline.named_steps
        assert 'preprocess' in pipeline.named_steps
    
    def test_create_preprocessing_pipeline_with_empty_columns(self):
        """Test pipeline creation with empty column lists."""
        pipeline = create_preprocessing_pipeline(
            columns_to_drop=[],
            numerical_columns=[],
            categorical_columns=[]
        )
        
        assert isinstance(pipeline, Pipeline)


class TestGetDefaultModels:
    """Test default model getter."""
    
    def test_get_default_models_returns_dict(self):
        """Test that function returns a dictionary."""
        models = get_default_models()
        
        assert isinstance(models, dict)
        assert len(models) == 3
    
    def test_get_default_models_contains_expected_models(self):
        """Test that all expected models are present."""
        models = get_default_models()
        
        assert 'Logistic Regression' in models
        assert 'Random Forest Classifier' in models
        assert 'Neural Network' in models
    
    def test_get_default_models_all_have_random_state(self):
        """Test that all models have random_state set."""
        models = get_default_models(random_state=123)
        
        for model_name, model in models.items():
            params = model.get_params()
            assert 'random_state' in params
            assert params['random_state'] == 123

class TestTrainAndEvaluateModel:
    """Test model training and evaluation."""
    
    def test_train_and_evaluate_model_without_mlflow(self, sample_train_data):
        """Test training without MLflow."""
        X_train, X_test, y_train, y_test, preprocess_pipeline = sample_train_data
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cm_path = os.path.join(tmpdir, "cm.png")
            
            pipeline, accuracy, report = train_and_evaluate_model(
                model=model,
                model_name="Test Model",
                preprocess_pipeline=preprocess_pipeline,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                use_mlflow=False,
                confusion_matrix_path=cm_path
            )
        
        assert isinstance(pipeline, Pipeline)
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1
        assert isinstance(report, str)
    
    @patch('src.models.train_model.mlflow')
    def test_train_and_evaluate_model_with_mlflow(self, mock_mlflow, sample_train_data):
        """Test training with MLflow logging."""
        X_train, X_test, y_train, y_test, preprocess_pipeline = sample_train_data
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cm_path = os.path.join(tmpdir, "cm.png")
            
            pipeline, accuracy, report = train_and_evaluate_model(
                model=model,
                model_name="Test Model",
                preprocess_pipeline=preprocess_pipeline,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                use_mlflow=True,
                confusion_matrix_path=cm_path
            )
        
        # Verify MLflow was called
        mock_mlflow.log_param.assert_called()
        mock_mlflow.log_params.assert_called()
        mock_mlflow.log_metric.assert_called()
        mock_mlflow.log_artifact.assert_called()
    
    def test_train_and_evaluate_model_returns_valid_pipeline(self, sample_train_data):
        """Test that returned pipeline can make predictions."""
        X_train, X_test, y_train, y_test, preprocess_pipeline = sample_train_data
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cm_path = os.path.join(tmpdir, "cm.png")
            
            pipeline, accuracy, report = train_and_evaluate_model(
                model=model,
                model_name="Test Model",
                preprocess_pipeline=preprocess_pipeline,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                use_mlflow=False,
                confusion_matrix_path=cm_path
            )
        
        # Test that pipeline can make predictions
        predictions = pipeline.predict(X_test)
        assert len(predictions) == len(y_test)
        assert set(predictions).issubset({0, 1})


class TestTrainModels:
    """Test training multiple models."""
    
    def test_train_models_without_mlflow(self, sample_train_data):
        """Test training multiple models without MLflow."""
        X_train, X_test, y_train, y_test, preprocess_pipeline = sample_train_data
        
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=10)
        }
        
        best_model, best_accuracy, best_model_name = train_models(
            models=models,
            preprocess_pipeline=preprocess_pipeline,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            use_mlflow=False
        )
        
        assert isinstance(best_model, Pipeline)
        assert isinstance(best_accuracy, float)
        assert 0 <= best_accuracy <= 1
        assert best_model_name in models.keys()
    
    @patch('src.models.train_model.mlflow')
    def test_train_models_with_mlflow(self, mock_mlflow, sample_train_data):
        """Test training multiple models with MLflow."""
        X_train, X_test, y_train, y_test, preprocess_pipeline = sample_train_data
        
        # Mock MLflow start_run to return a context manager
        mock_mlflow.start_run.return_value.__enter__ = Mock()
        mock_mlflow.start_run.return_value.__exit__ = Mock()
        
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        best_model, best_accuracy, best_model_name = train_models(
            models=models,
            preprocess_pipeline=preprocess_pipeline,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            use_mlflow=True,
            experiment_name="test_experiment"
        )
        
        # Verify MLflow was configured
        mock_mlflow.set_experiment.assert_called_once_with("test_experiment")
    
    def test_train_models_returns_best_model(self, sample_train_data):
        """Test that the function returns the model with highest accuracy."""
        X_train, X_test, y_train, y_test, preprocess_pipeline = sample_train_data
        
        models = {
            'Model1': LogisticRegression(random_state=42, max_iter=1000),
            'Model2': LogisticRegression(random_state=43, max_iter=1000)
        }
        
        best_model, best_accuracy, best_model_name = train_models(
            models=models,
            preprocess_pipeline=preprocess_pipeline,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            use_mlflow=False
        )
        
        assert best_model_name in models.keys()


class TestSaveModel:
    """Test model saving."""
    
    def test_save_model_creates_file(self):
        """Test that model file is created."""
        model = Pipeline([
            ('regressor', LogisticRegression())
        ])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_model.pkl")
            save_model(model, output_path)
            
            assert os.path.exists(output_path)
    
    def test_save_model_can_be_loaded(self):
        """Test that saved model can be loaded."""
        model = Pipeline([
            ('regressor', LogisticRegression())
        ])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_model.pkl")
            save_model(model, output_path)
            
            loaded_model = joblib.load(output_path)
            assert isinstance(loaded_model, Pipeline)

