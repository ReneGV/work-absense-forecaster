"""
Tests for prediction functionality in src/models/predict_model.py
"""
import pytest
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from src.models.preprocessors import (
    DropColumnsTransformer,
    IQRClippingTransformer,
    ToStringTransformer
)
import tempfile
import os


class TestModelLoading:
    """Tests for model loading functionality"""

    def test_model_can_be_saved_and_loaded(self, sample_dataframe, columns_to_drop,
                                          numerical_columns, categorical_columns):
        """Test that a trained model can be saved and loaded"""
        # Train a simple model
        preprocess_pipeline = Pipeline([
            ('drop_columns', DropColumnsTransformer(columns_to_drop)),
            ('preprocess', ColumnTransformer(
                transformers=[
                    ('numerical', Pipeline([
                        ('iqr_clipping', IQRClippingTransformer()),
                        ('scaling', StandardScaler())
                    ]), numerical_columns),
                    ('categorical', Pipeline([
                        ('to_string', ToStringTransformer()),
                        ('one_hot', OneHotEncoder(sparse=False, handle_unknown='ignore'))
                    ]), categorical_columns)
                ],
                remainder='passthrough'
            ))
        ])
        
        full_pipeline = Pipeline([
            ('preprocess', preprocess_pipeline),
            ('regressor', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        X = sample_dataframe.drop('absenteeism_time_in_hours', axis=1)
        y = sample_dataframe['absenteeism_time_in_hours']
        y = (y > y.median()).astype(int)
        
        full_pipeline.fit(X, y)
        
        # Save model to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            joblib.dump(full_pipeline, tmp.name)
            tmp_path = tmp.name
        
        try:
            # Load model
            loaded_model = joblib.load(tmp_path)
            
            # Test that loaded model works
            predictions = loaded_model.predict(X)
            
            assert predictions is not None
            assert len(predictions) == len(y)
        finally:
            # Cleanup
            os.unlink(tmp_path)


class TestPredictionProcess:
    """Tests for the prediction process"""

    def test_prediction_on_new_data(self, sample_dataframe, columns_to_drop,
                                   numerical_columns, categorical_columns):
        """Test making predictions on new data"""
        # Train model
        preprocess_pipeline = Pipeline([
            ('drop_columns', DropColumnsTransformer(columns_to_drop)),
            ('preprocess', ColumnTransformer(
                transformers=[
                    ('numerical', Pipeline([
                        ('iqr_clipping', IQRClippingTransformer()),
                        ('scaling', StandardScaler())
                    ]), numerical_columns),
                    ('categorical', Pipeline([
                        ('to_string', ToStringTransformer()),
                        ('one_hot', OneHotEncoder(sparse=False, handle_unknown='ignore'))
                    ]), categorical_columns)
                ],
                remainder='passthrough'
            ))
        ])
        
        full_pipeline = Pipeline([
            ('preprocess', preprocess_pipeline),
            ('regressor', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        X = sample_dataframe.drop('absenteeism_time_in_hours', axis=1)
        y = sample_dataframe['absenteeism_time_in_hours']
        y = (y > y.median()).astype(int)
        
        full_pipeline.fit(X, y)
        
        # Create new data (sample from same dataframe)
        new_data = X.sample(3, random_state=42)
        
        # Make predictions
        predictions = full_pipeline.predict(new_data)
        
        assert len(predictions) == 3
        assert all(pred in [0, 1] for pred in predictions)

    def test_prediction_labels_conversion(self):
        """Test converting numeric predictions to labels"""
        predictions = np.array([0, 1, 1, 0, 1])
        prediction_labels = ['High' if p == 1 else 'Low' for p in predictions]
        
        assert len(prediction_labels) == len(predictions)
        assert prediction_labels == ['Low', 'High', 'High', 'Low', 'High']
        assert all(label in ['High', 'Low'] for label in prediction_labels)


class TestPredictionEvaluation:
    """Tests for evaluating predictions"""

    def test_evaluation_with_ground_truth(self):
        """Test evaluating predictions when ground truth is available"""
        from sklearn.metrics import accuracy_score, f1_score, recall_score
        
        # Simulate predictions and ground truth
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 1])
        
        # Calculate metrics
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        
        assert 0 <= acc <= 1
        assert 0 <= f1 <= 1
        assert 0 <= recall <= 1

    def test_median_threshold_binary_conversion(self):
        """Test converting continuous values to binary using median threshold"""
        values = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        median_value = values.median()
        binary_values = (values > median_value).astype(int)
        
        # Check that conversion works
        assert binary_values.sum() == 5  # Half should be above median
        assert len(binary_values) == len(values)
        assert set(binary_values.unique()).issubset({0, 1})

    def test_confusion_matrix_with_predictions(self):
        """Test creating confusion matrix with predictions"""
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 1])
        
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'High'])
        
        assert cm.shape == (2, 2)
        assert disp is not None


class TestDataHandling:
    """Tests for data handling in predictions"""

    def test_drop_columns_with_errors_ignore(self, sample_dataframe):
        """Test dropping columns with errors='ignore' parameter"""
        df = sample_dataframe.copy()
        
        # Try to drop a column that exists and one that doesn't
        result = df.drop(columns=['absenteeism_time_in_hours', 'nonexistent_col'], errors='ignore')
        
        assert 'absenteeism_time_in_hours' not in result.columns
        assert result.shape[0] == df.shape[0]

    def test_adding_predictions_to_dataframe(self, sample_dataframe):
        """Test adding prediction column to dataframe"""
        df = sample_dataframe.copy()
        predictions = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        prediction_labels = ['High' if p == 1 else 'Low' for p in predictions]
        
        df['predicted_absenteeism'] = prediction_labels
        
        assert 'predicted_absenteeism' in df.columns
        assert len(df['predicted_absenteeism']) == len(df)
        assert all(label in ['High', 'Low'] for label in df['predicted_absenteeism'])

    def test_sampling_dataframe(self, sample_dataframe):
        """Test sampling from dataframe"""
        sample_size = 5
        sampled = sample_dataframe.sample(sample_size, random_state=42)
        
        assert len(sampled) == sample_size
        assert all(col in sample_dataframe.columns for col in sampled.columns)


class TestPredictionMetrics:
    """Tests for prediction metrics"""

    def test_all_metrics_calculation(self):
        """Test calculating all required metrics"""
        from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report
        
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 1, 0, 1])
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        
        # All metrics should be calculated successfully
        assert acc is not None
        assert f1 is not None
        assert recall is not None
        assert report is not None
        
        # Metrics should be in valid range
        assert 0 <= acc <= 1
        assert 0 <= f1 <= 1
        assert 0 <= recall <= 1

    def test_f1_score_edge_cases(self):
        """Test F1 score with edge cases"""
        from sklearn.metrics import f1_score
        
        # Perfect predictions
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])
        f1 = f1_score(y_true, y_pred)
        assert f1 == 1.0
        
        # All wrong (for class 1)
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 1])
        f1 = f1_score(y_true, y_pred, zero_division=0)
        assert f1 == 0.0

