"""
Tests for training model functionality in src/models/train_model.py
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from src.models.preprocessors import (
    DropColumnsTransformer,
    IQRClippingTransformer,
    ToStringTransformer
)


class TestPipelineConstruction:
    """Tests for pipeline construction used in train_model.py"""

    def test_preprocess_pipeline_creation(self, columns_to_drop, numerical_columns, categorical_columns):
        """Test that preprocessing pipeline can be created"""
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
        
        assert preprocess_pipeline is not None
        assert len(preprocess_pipeline.steps) == 2

    def test_full_pipeline_with_model(self, sample_dataframe, columns_to_drop, 
                                     numerical_columns, categorical_columns):
        """Test that full pipeline with model can be created and fitted"""
        # Create preprocessing pipeline
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
        
        # Create full pipeline with model
        full_pipeline = Pipeline([
            ('preprocess', preprocess_pipeline),
            ('regressor', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        # Prepare data
        X = sample_dataframe.drop('absenteeism_time_in_hours', axis=1)
        y = sample_dataframe['absenteeism_time_in_hours']
        y = (y > y.median()).astype(int)
        
        # Fit pipeline
        full_pipeline.fit(X, y)
        
        # Make predictions
        predictions = full_pipeline.predict(X)
        
        assert predictions is not None
        assert len(predictions) == len(y)
        assert all(pred in [0, 1] for pred in predictions)


class TestModelCreation:
    """Tests for model instantiation"""

    def test_logistic_regression_creation(self):
        """Test Logistic Regression model creation"""
        model = LogisticRegression(random_state=42)
        assert model is not None
        assert model.random_state == 42

    def test_random_forest_creation(self):
        """Test Random Forest model creation"""
        model = RandomForestClassifier(random_state=42, n_estimators=100)
        assert model is not None
        assert model.random_state == 42
        assert model.n_estimators == 100

    def test_neural_network_creation(self):
        """Test Neural Network model creation"""
        model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.01,
            max_iter=1000,
            random_state=42,
            early_stopping=True
        )
        assert model is not None
        assert model.random_state == 42
        assert model.hidden_layer_sizes == (100, 50)
        assert model.early_stopping == True


class TestDataPreparation:
    """Tests for data preparation steps"""

    def test_column_name_normalization(self, sample_dataframe):
        """Test that column names are normalized correctly"""
        df = sample_dataframe.copy()
        df.columns = df.columns.str.lower().str.replace("[ ]", "_", regex=True)
        
        # Check that all columns are lowercase and have no spaces
        for col in df.columns:
            assert col == col.lower()
            assert ' ' not in col

    def test_binary_target_creation(self, sample_dataframe):
        """Test that binary target is created correctly"""
        y = sample_dataframe['absenteeism_time_in_hours']
        median_value = y.median()
        y_binary = (y > median_value).astype(int)
        
        # Check that target is binary
        assert set(y_binary.unique()).issubset({0, 1})
        
        # Check that roughly half are 0 and half are 1
        assert 0 < y_binary.sum() < len(y_binary)

    def test_train_test_split_compatibility(self, sample_dataframe):
        """Test that data can be split for training and testing"""
        from sklearn.model_selection import train_test_split
        
        X = sample_dataframe.drop('absenteeism_time_in_hours', axis=1)
        y = sample_dataframe['absenteeism_time_in_hours']
        y = (y > y.median()).astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Check splits are correct
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert len(X_train) + len(X_test) == len(X)


class TestModelTraining:
    """Tests for model training process"""

    def test_logistic_regression_training(self, sample_dataframe, columns_to_drop,
                                         numerical_columns, categorical_columns):
        """Test that Logistic Regression can be trained"""
        # Prepare pipeline
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
        
        # Prepare data
        X = sample_dataframe.drop('absenteeism_time_in_hours', axis=1)
        y = sample_dataframe['absenteeism_time_in_hours']
        y = (y > y.median()).astype(int)
        
        # Train
        full_pipeline.fit(X, y)
        
        # Check model is fitted
        assert hasattr(full_pipeline.named_steps['regressor'], 'coef_')

    def test_model_prediction_output_format(self, sample_dataframe, columns_to_drop,
                                           numerical_columns, categorical_columns):
        """Test that model predictions have correct format"""
        # Quick pipeline
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
        predictions = full_pipeline.predict(X)
        
        # Check predictions format
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)
        assert all(pred in [0, 1] for pred in predictions)


class TestMetricsCalculation:
    """Tests for metrics calculation"""

    def test_accuracy_score_calculation(self):
        """Test accuracy score calculation"""
        from sklearn.metrics import accuracy_score
        
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        accuracy = accuracy_score(y_true, y_pred)
        
        assert 0 <= accuracy <= 1
        assert accuracy == 0.8  # 4 out of 5 correct

    def test_confusion_matrix_creation(self):
        """Test confusion matrix creation"""
        from sklearn.metrics import confusion_matrix
        
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        cm = confusion_matrix(y_true, y_pred)
        
        assert cm.shape == (2, 2)
        assert cm.sum() == len(y_true)

    def test_classification_report_creation(self):
        """Test classification report creation"""
        from sklearn.metrics import classification_report
        
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        report = classification_report(y_true, y_pred)
        
        assert report is not None
        assert isinstance(report, str)
        assert 'precision' in report
        assert 'recall' in report

