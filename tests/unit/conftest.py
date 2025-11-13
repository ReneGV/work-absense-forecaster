"""
Shared fixtures for unit tests.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from sklearn.pipeline import Pipeline

from src.models.train_model import create_preprocessing_pipeline
from src.data.constants import NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS


@pytest.fixture
def sample_csv_file():
    """Create a sample CSV file for testing."""
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'ID': range(1, n_samples + 1),
        'Reason for absence': np.random.randint(0, 28, n_samples),
        'Month of absence': np.random.randint(0, 12, n_samples),
        'Day of the week': np.random.randint(2, 6, n_samples),
        'Seasons': np.random.randint(1, 5, n_samples),
        'Transportation expense': np.random.randint(100, 400, n_samples),
        'Distance from Residence to Work': np.random.randint(5, 50, n_samples),
        'Service time': np.random.randint(1, 30, n_samples),
        'Age': np.random.randint(18, 60, n_samples),
        'Work load Average/day': np.random.uniform(200000, 400000, n_samples),
        'Hit target': np.random.randint(80, 100, n_samples),
        'Disciplinary failure': np.random.randint(0, 2, n_samples),
        'Education': np.random.randint(1, 5, n_samples),
        'Son': np.random.randint(0, 4, n_samples),
        'Social drinker': np.random.randint(0, 2, n_samples),
        'Social smoker': np.random.randint(0, 2, n_samples),
        'Pet': np.random.randint(0, 8, n_samples),
        'Weight': np.random.randint(60, 100, n_samples),
        'Height': np.random.randint(160, 190, n_samples),
        'Body mass index': np.random.randint(18, 35, n_samples),
        'Absenteeism time in hours': np.random.randint(0, 40, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_train_data():
    """Create sample training data."""
    np.random.seed(42)
    n_samples = 100
    
    # Create sample DataFrame
    data = {
        'id': range(1, n_samples + 1),
        'reason_for_absence': np.random.randint(0, 28, n_samples),
        'month_of_absence': np.random.randint(0, 12, n_samples),
        'day_of_the_week': np.random.randint(2, 6, n_samples),
        'seasons': np.random.randint(1, 5, n_samples),
        'transportation_expense': np.random.randint(100, 400, n_samples),
        'distance_from_residence_to_work': np.random.randint(5, 50, n_samples),
        'service_time': np.random.randint(1, 30, n_samples),
        'age': np.random.randint(18, 60, n_samples),
        'work_load_average/day': np.random.uniform(200000, 400000, n_samples),
        'hit_target': np.random.randint(80, 100, n_samples),
        'disciplinary_failure': np.random.randint(0, 2, n_samples),
        'education': np.random.randint(1, 5, n_samples),
        'son': np.random.randint(0, 4, n_samples),
        'social_drinker': np.random.randint(0, 2, n_samples),
        'social_smoker': np.random.randint(0, 2, n_samples),
        'pet': np.random.randint(0, 8, n_samples),
        'weight': np.random.randint(60, 100, n_samples),
        'height': np.random.randint(160, 190, n_samples),
        'body_mass_index': np.random.randint(18, 35, n_samples),
    }
    
    X = pd.DataFrame(data)
    y = pd.Series(np.random.randint(0, 2, n_samples))
    
    # Split into train and test
    split_idx = int(0.8 * n_samples)
    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()
    
    # Create preprocessing pipeline
    columns_to_drop = ['id', 'body_mass_index']
    
    preprocess_pipeline = create_preprocessing_pipeline(
        columns_to_drop,
        NUMERICAL_COLUMNS,
        CATEGORICAL_COLUMNS
    )
    
    return X_train, X_test, y_train, y_test, preprocess_pipeline

