"""
Pytest fixtures for model tests
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing"""
    np.random.seed(42)
    return pd.DataFrame({
        'id': range(1, 11),
        'body_mass_index': np.random.uniform(18, 35, 10),
        'transportation_expense': np.random.uniform(100, 300, 10),
        'distance_from_residence_to_work': np.random.uniform(5, 50, 10),
        'service_time': np.random.randint(1, 20, 10),
        'age': np.random.randint(25, 60, 10),
        'work_load_average/day': np.random.uniform(200, 400, 10),
        'hit_target': np.random.randint(80, 100, 10),
        'son': np.random.randint(0, 3, 10),
        'pet': np.random.randint(0, 5, 10),
        'weight': np.random.uniform(50, 100, 10),
        'height': np.random.uniform(150, 190, 10),
        'education': np.random.randint(1, 5, 10),
        'disciplinary_failure': np.random.choice([0, 1], 10),
        'social_drinker': np.random.choice([0, 1], 10),
        'social_smoker': np.random.choice([0, 1], 10),
        'month_of_absence': np.random.randint(0, 13, 10),
        'day_of_the_week': np.random.randint(2, 7, 10),
        'seasons': np.random.randint(1, 5, 10),
        'reason_for_absence': np.random.randint(0, 29, 10),
        'absenteeism_time_in_hours': np.random.randint(0, 40, 10)
    })


@pytest.fixture
def numerical_dataframe():
    """Create a DataFrame with numerical data for testing"""
    np.random.seed(42)
    return pd.DataFrame({
        'col1': [1, 2, 3, 4, 5, 100],  # 100 is an outlier
        'col2': [10, 20, 30, 40, 50, 60],
        'col3': [0.1, 0.2, 0.3, 0.4, 0.5, 1000]  # 1000 is an outlier
    })


@pytest.fixture
def categorical_dataframe():
    """Create a DataFrame with categorical data for testing"""
    return pd.DataFrame({
        'cat1': [1, 2, 1, 2, 3],
        'cat2': ['A', 'B', 'A', 'C', 'B'],
        'cat3': [10, 20, 10, 30, 20]
    })


@pytest.fixture
def columns_to_drop():
    """List of columns to drop"""
    return ['id', 'body_mass_index']


@pytest.fixture
def numerical_columns():
    """List of numerical columns"""
    return [
        'transportation_expense', 'distance_from_residence_to_work', 'service_time',
        'age', 'work_load_average/day', 'hit_target', 'son', 'pet', 'weight',
        'height', 'education'
    ]


@pytest.fixture
def categorical_columns():
    """List of categorical columns"""
    return [
        'disciplinary_failure', 'social_drinker', 'social_smoker',
        'month_of_absence', 'day_of_the_week', 'seasons', 'reason_for_absence'
    ]

