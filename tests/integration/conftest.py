"""
Shared fixtures for integration tests.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.models.train_model import (
    create_preprocessing_pipeline,
    load_and_prepare_data
)
from src.models.train_model import save_model
from src.data.constants import NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS


@pytest.fixture
def sample_data_file():
    """Create a temporary CSV file with sample absenteeism data for testing."""
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
def sample_dataframe():
    """Create a sample DataFrame for testing preprocessors."""
    np.random.seed(42)
    return pd.DataFrame({
        'id': range(1, 51),
        'body_mass_index': np.random.uniform(18, 35, 50),
        'transportation_expense': np.random.uniform(100, 300, 50),
        'distance_from_residence_to_work': np.random.uniform(5, 50, 50),
        'service_time': np.random.randint(1, 20, 50),
        'age': np.random.randint(25, 60, 50),
        'work_load_average/day': np.random.uniform(200000, 400000, 50),
        'hit_target': np.random.randint(80, 100, 50),
        'son': np.random.randint(0, 3, 50),
        'pet': np.random.randint(0, 5, 50),
        'weight': np.random.uniform(50, 100, 50),
        'height': np.random.uniform(150, 190, 50),
        'education': np.random.randint(1, 5, 50),
        'disciplinary_failure': np.random.choice([0, 1], 50),
        'social_drinker': np.random.choice([0, 1], 50),
        'social_smoker': np.random.choice([0, 1], 50),
        'month_of_absence': np.random.randint(0, 13, 50),
        'day_of_the_week': np.random.randint(2, 7, 50),
        'seasons': np.random.randint(1, 5, 50),
        'reason_for_absence': np.random.randint(0, 29, 50),
        'absenteeism_time_in_hours': np.random.randint(0, 40, 50)
    })


@pytest.fixture
def trained_model(sample_data_file):
    """Create and return a trained model for testing predictions."""
    # Load and prepare data
    X_train, X_test, y_train, y_test, _ = load_and_prepare_data(
        sample_data_file,
        test_size=0.3,
        random_state=42
    )
    
    # Create preprocessing pipeline
    columns_to_drop = ['id', 'body_mass_index']
    
    preprocess_pipeline = create_preprocessing_pipeline(
        columns_to_drop,
        NUMERICAL_COLUMNS,
        CATEGORICAL_COLUMNS
    )
    
    # Create and train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    full_pipeline = Pipeline([
        ('preprocess', preprocess_pipeline),
        ('regressor', model)
    ])
    
    full_pipeline.fit(X_train, y_train)
    
    return full_pipeline

