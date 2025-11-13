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
from src.models.predict_model import (
    load_model,
    prepare_test_data,
    make_predictions,
    convert_to_labels,
    save_predictions,
    convert_to_binary
)
import tempfile
import os


class TestModelLoading:
    """Tests for model loading functionality"""

    def test_load_model_function(self, sample_dataframe, columns_to_drop,
                                 numerical_columns, categorical_columns):
        """Test the load_model function"""
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
            # Load model using the function
            loaded_model = load_model(tmp_path)
            
            # Test that loaded model works
            predictions = loaded_model.predict(X)
            
            assert predictions is not None
            assert len(predictions) == len(y)
            assert all(pred in [0, 1] for pred in predictions)
        finally:
            # Cleanup
            os.unlink(tmp_path)


class TestPredictionProcess:
    """Tests for the prediction process"""

    def test_make_predictions_function(self, sample_dataframe, columns_to_drop,
                                      numerical_columns, categorical_columns):
        """Test the make_predictions function"""
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
        
        # Make predictions using the function
        predictions = make_predictions(full_pipeline, new_data, exclude_columns=[])
        
        assert len(predictions) == 3
        assert all(pred in [0, 1] for pred in predictions)

    def test_convert_to_labels_function(self):
        """Test the convert_to_labels function"""
        predictions = np.array([0, 1, 1, 0, 1])
        prediction_labels = convert_to_labels(predictions)
        
        assert len(prediction_labels) == len(predictions)
        assert prediction_labels == ['Low', 'High', 'High', 'Low', 'High']
        assert all(label in ['High', 'Low'] for label in prediction_labels)
    

class TestDataHandling:
    """Tests for data handling in predictions"""

    def test_prepare_test_data_function(self, sample_dataframe):
        """Test the prepare_test_data function"""
        df = sample_dataframe.copy()
        
        # Rename column to match expected format
        if 'absenteeism_time_in_hours' in df.columns:
            sample_size = 5
            result = prepare_test_data(df, sample_size=sample_size, random_state=42)
            
            # Check that data was sampled
            assert len(result) == sample_size
            
            # Check that ground truth was preserved
            if 'absenteeism_time_in_hours' in df.columns:
                assert 'absenteeism_real' in result.columns
                assert 'absenteeism_time_in_hours' not in result.columns
    
    def test_save_predictions_function(self, sample_dataframe):
        """Test the save_predictions function"""
        df = sample_dataframe.copy()
        predictions = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        prediction_labels = convert_to_labels(predictions)
        
        df['predicted_absenteeism'] = prediction_labels
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w') as tmp:
            tmp_path = tmp.name
        
        try:
            save_predictions(df, tmp_path)
            
            # Check that file was created and can be read
            assert os.path.exists(tmp_path)
            loaded_df = pd.read_csv(tmp_path)
            assert 'predicted_absenteeism' in loaded_df.columns
            assert len(loaded_df) == len(df)
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)



