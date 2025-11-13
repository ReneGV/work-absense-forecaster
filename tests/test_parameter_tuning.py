"""
Tests for hyperparameter tuning functionality in src/models/parameter_tuning.py
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, ParameterSampler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.models.preprocessors import (
    DropColumnsTransformer,
    IQRClippingTransformer,
    ToStringTransformer
)


class TestParameterGrid:
    """Tests for parameter grid definition"""

    def test_parameter_grid_structure(self):
        """Test that parameter grid has correct structure"""
        param_dist = {
            'clf__C': [0.01, 0.1, 1, 10],
            'clf__fit_intercept': [True, False],
            'clf__class_weight': [None, 'balanced']
        }
        
        assert 'clf__C' in param_dist
        assert 'clf__fit_intercept' in param_dist
        assert 'clf__class_weight' in param_dist
        assert len(param_dist['clf__C']) == 4
        assert len(param_dist['clf__fit_intercept']) == 2

    def test_parameter_sampler(self):
        """Test parameter sampling functionality"""
        param_dist = {
            'clf__C': [0.01, 0.1, 1, 10],
            'clf__fit_intercept': [True, False],
            'clf__class_weight': [None, 'balanced']
        }
        
        param_grid = list(ParameterSampler(param_dist, n_iter=16, random_state=42))
        
        assert len(param_grid) == 16
        assert all('clf__C' in params for params in param_grid)
        assert all('clf__fit_intercept' in params for params in param_grid)


class TestGridSearchSetup:
    """Tests for GridSearchCV setup"""

    def test_grid_search_initialization(self, sample_dataframe, columns_to_drop,
                                       numerical_columns, categorical_columns):
        """Test that GridSearchCV can be initialized"""
        # Create pipeline
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
        
        pipe = Pipeline([
            ("preprocess", preprocess_pipeline),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ])
        
        param_dist = {
            'clf__C': [0.1, 1],
            'clf__fit_intercept': [True, False],
        }
        
        search = GridSearchCV(
            pipe, param_dist, cv=3, scoring="f1", n_jobs=1, verbose=0
        )
        
        assert search is not None
        assert search.cv == 3
        assert search.scoring == "f1"

    def test_grid_search_fit(self, sample_dataframe, columns_to_drop,
                            numerical_columns, categorical_columns):
        """Test that GridSearchCV can fit on data"""
        # Create pipeline
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
        
        pipe = Pipeline([
            ("preprocess", preprocess_pipeline),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ])
        
        param_dist = {
            'clf__C': [0.1, 1],
        }
        
        search = GridSearchCV(
            pipe, param_dist, cv=2, scoring="f1", n_jobs=1, verbose=0
        )
        
        X = sample_dataframe.drop('absenteeism_time_in_hours', axis=1)
        y = sample_dataframe['absenteeism_time_in_hours']
        y = (y > y.median()).astype(int)
        
        search.fit(X, y)
        
        assert hasattr(search, 'best_estimator_')
        assert hasattr(search, 'best_params_')
        assert hasattr(search, 'best_score_')


class TestModelTuning:
    """Tests for model tuning process"""

    def test_pipeline_set_params(self, columns_to_drop, numerical_columns, categorical_columns):
        """Test that pipeline parameters can be set"""
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
        
        pipe = Pipeline([
            ("preprocess", preprocess_pipeline),
            ("clf", LogisticRegression(max_iter=1000))
        ])
        
        # Set parameters
        params = {'clf__C': 0.5, 'clf__fit_intercept': False}
        pipe.set_params(**params)
        
        # Check parameters were set
        assert pipe.named_steps['clf'].C == 0.5
        assert pipe.named_steps['clf'].fit_intercept == False

    def test_multiple_parameter_combinations(self, sample_dataframe, columns_to_drop,
                                            numerical_columns, categorical_columns):
        """Test training with multiple parameter combinations"""
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
        
        pipe = Pipeline([
            ("preprocess", preprocess_pipeline),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ])
        
        X = sample_dataframe.drop('absenteeism_time_in_hours', axis=1)
        y = sample_dataframe['absenteeism_time_in_hours']
        y = (y > y.median()).astype(int)
        
        # Test multiple parameter combinations
        param_combinations = [
            {'clf__C': 0.1, 'clf__fit_intercept': True},
            {'clf__C': 1.0, 'clf__fit_intercept': False},
        ]
        
        results = []
        for params in param_combinations:
            pipe.set_params(**params)
            pipe.fit(X, y)
            score = pipe.score(X, y)
            results.append(score)
        
        assert len(results) == 2
        assert all(0 <= score <= 1 for score in results)


class TestBestModelSelection:
    """Tests for selecting best model"""

    def test_best_model_tracking(self, sample_dataframe, columns_to_drop,
                                 numerical_columns, categorical_columns):
        """Test tracking the best model during tuning"""
        from sklearn.metrics import f1_score
        
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
        
        pipe = Pipeline([
            ("preprocess", preprocess_pipeline),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ])
        
        X = sample_dataframe.drop('absenteeism_time_in_hours', axis=1)
        y = sample_dataframe['absenteeism_time_in_hours']
        y = (y > y.median()).astype(int)
        
        # Simulate tracking best model
        best_f1 = -1.0
        best_params = None
        best_model = None
        
        param_combinations = [
            {'clf__C': 0.1},
            {'clf__C': 1.0},
        ]
        
        for params in param_combinations:
            pipe.set_params(**params)
            pipe.fit(X, y)
            y_pred = pipe.predict(X)
            f1 = f1_score(y, y_pred)
            
            if f1 > best_f1:
                best_f1 = f1
                best_params = params
                best_model = pipe
        
        assert best_f1 >= 0
        assert best_params is not None
        assert best_model is not None

    def test_best_estimator_attributes(self, sample_dataframe, columns_to_drop,
                                      numerical_columns, categorical_columns):
        """Test that best estimator has required attributes"""
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
        
        pipe = Pipeline([
            ("preprocess", preprocess_pipeline),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ])
        
        param_dist = {
            'clf__C': [0.1, 1],
        }
        
        search = GridSearchCV(
            pipe, param_dist, cv=2, scoring="f1", n_jobs=1, verbose=0
        )
        
        X = sample_dataframe.drop('absenteeism_time_in_hours', axis=1)
        y = sample_dataframe['absenteeism_time_in_hours']
        y = (y > y.median()).astype(int)
        
        search.fit(X, y)
        best = search.best_estimator_
        
        # Check that best estimator can make predictions
        predictions = best.predict(X)
        assert len(predictions) == len(y)
        
        # Check that best estimator has required methods
        assert hasattr(best, 'predict')
        assert hasattr(best, 'fit')
        assert hasattr(best, 'score')


class TestMetricsTracking:
    """Tests for metrics tracking during tuning"""

    def test_train_test_metrics(self, sample_dataframe, columns_to_drop,
                                numerical_columns, categorical_columns):
        """Test calculating train and test metrics"""
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score, accuracy_score
        
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
        
        pipe = Pipeline([
            ("preprocess", preprocess_pipeline),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ])
        
        X = sample_dataframe.drop('absenteeism_time_in_hours', axis=1)
        y = sample_dataframe['absenteeism_time_in_hours']
        y = (y > y.median()).astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        pipe.fit(X_train, y_train)
        
        y_pred_train = pipe.predict(X_train)
        y_pred_test = pipe.predict(X_test)
        
        train_f1 = f1_score(y_train, y_pred_train)
        test_f1 = f1_score(y_test, y_pred_test)
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        # All metrics should be valid
        assert 0 <= train_f1 <= 1
        assert 0 <= test_f1 <= 1
        assert 0 <= train_acc <= 1
        assert 0 <= test_acc <= 1

