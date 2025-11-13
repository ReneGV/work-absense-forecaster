"""
Train models for absenteeism prediction with MLflow tracking.

This module provides functions to:
- Load and prepare data for training
- Create preprocessing pipelines
- Train multiple models with MLflow tracking
- Evaluate and save the best model
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, Tuple, Optional, List
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from src.models.preprocessors import DropColumnsTransformer, IQRClippingTransformer, ToStringTransformer
from src.models.evaluation import evaluate_predictions, get_classification_report, create_confusion_matrix
from src.data.constants import NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS
import mlflow

from dotenv import load_dotenv
import os


def load_and_prepare_data(
    file_path: str,
    target_column: str = 'absenteeism_time_in_hours',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, float]:
    """
    Load data from CSV and prepare it for training.
    
    Args:
        file_path: Path to the CSV file
        target_column: Name of the target column
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple containing (X_train, X_test, y_train, y_test, median_value)
    """
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.lower().str.replace("[ ]", "_", regex=True)
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Convert to binary classification based on median
    median_value = y.median()
    y = (y > median_value).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, median_value


def create_preprocessing_pipeline(
    columns_to_drop: List[str],
    numerical_columns: List[str],
    categorical_columns: List[str]
) -> Pipeline:
    """
    Create a preprocessing pipeline for the data.
    
    Args:
        columns_to_drop: List of column names to drop
        numerical_columns: List of numerical column names
        categorical_columns: List of categorical column names
        
    Returns:
        sklearn Pipeline for preprocessing
    """
    return Pipeline([
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


def get_default_models(random_state: int = 42) -> Dict[str, BaseEstimator]:
    """
    Get a dictionary of default models to train.
    
    Args:
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary mapping model names to model instances
    """
    return {
        'Logistic Regression': LogisticRegression(random_state=random_state),
        'Random Forest Classifier': RandomForestClassifier(
            random_state=random_state,
            n_estimators=100
        ),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.01,
            max_iter=1000,
            random_state=random_state,
            early_stopping=True
        )
    }


def train_and_evaluate_model(
    model: BaseEstimator,
    model_name: str,
    preprocess_pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    use_mlflow: bool = True,
    confusion_matrix_path: str = "confusion_matrix.png"
) -> Tuple[Pipeline, float, str]:
    """
    Train and evaluate a single model.
    
    Args:
        model: The model to train
        model_name: Name of the model
        preprocess_pipeline: Preprocessing pipeline
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        use_mlflow: Whether to log to MLflow
        confusion_matrix_path: Path to save confusion matrix
        
    Returns:
        Tuple containing (trained_pipeline, accuracy, classification_report)
    """
    full_pipeline = Pipeline([
        ('preprocess', preprocess_pipeline),
        ('regressor', model)
    ])
    
    # Training
    full_pipeline.fit(X_train, y_train)
    y_pred = full_pipeline.predict(X_test)
    
    # Evaluate using evaluation.py functions
    metrics = evaluate_predictions(y_test, y_pred)
    report = get_classification_report(y_test, y_pred)
    
    if use_mlflow:
        # Log parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_params(model.get_params())
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Create and log confusion matrix using evaluation.py
        plot_path = create_confusion_matrix(
            y_test, 
            y_pred, 
            confusion_matrix_path,
            title=f'{model_name}: Confusion Matrix'
        )
        mlflow.log_artifact(plot_path)
    
    print(f"\nResults for {model_name}:")
    print(f"Accuracy: {metrics['accuracy']:.2f}")
    print(f"F1 Score: {metrics['f1_score']:.2f}")
    print(f"Recall: {metrics['recall']:.2f}")
    print("Classification Report:")
    print(report)
    
    return full_pipeline, metrics['accuracy'], report


def train_models(
    models: Dict[str, BaseEstimator],
    preprocess_pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    use_mlflow: bool = True,
    experiment_name: Optional[str] = None
) -> Tuple[Pipeline, float, str]:
    """
    Train multiple models and return the best one.
    
    Args:
        models: Dictionary of models to train
        preprocess_pipeline: Preprocessing pipeline
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        use_mlflow: Whether to use MLflow tracking
        experiment_name: Name of MLflow experiment
        
    Returns:
        Tuple containing (best_model, best_accuracy, best_model_name)
    """
    if use_mlflow and experiment_name:
        mlflow.set_experiment(experiment_name)
    
    best_model = None
    best_accuracy = 0
    best_model_name = None
    
    for model_name, model in models.items():
        if use_mlflow:
            with mlflow.start_run(run_name=model_name):
                pipeline, accuracy, _ = train_and_evaluate_model(
                    model, model_name, preprocess_pipeline,
                    X_train, X_test, y_train, y_test,
                    use_mlflow=True
                )
                
                # Track best model
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = pipeline
                    best_model_name = model_name
                    mlflow.sklearn.log_model(best_model, artifact_path="best_model")
        else:
            pipeline, accuracy, _ = train_and_evaluate_model(
                model, model_name, preprocess_pipeline,
                X_train, X_test, y_train, y_test,
                use_mlflow=False
            )
            
            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = pipeline
                best_model_name = model_name
    
    return best_model, best_accuracy, best_model_name


def save_model(model: Pipeline, output_path: str) -> None:
    """
    Save a trained model to disk.
    
    Args:
        model: The trained model pipeline
        output_path: Path where to save the model
    """
    joblib.dump(model, output_path)


def main():
    """Main training function to run the complete training pipeline."""
    # Load environment and configure MLflow
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    
    # Define column configurations
    columns_to_drop = ['id', 'body_mass_index']
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, median_value = load_and_prepare_data(
        'data/raw/work_absenteeism_original.csv'
    )
    
    # Create preprocessing pipeline
    preprocess_pipeline = create_preprocessing_pipeline(
        columns_to_drop,
        NUMERICAL_COLUMNS,
        CATEGORICAL_COLUMNS
    )
    
    # Get models to train
    models = get_default_models()
    
    # Train models
    best_model, best_accuracy, best_model_name = train_models(
        models,
        preprocess_pipeline,
        X_train, X_test, y_train, y_test,
        use_mlflow=True,
        experiment_name="absenteeism_forecasting_best_model"
    )
    
    # Save best model
    save_model(best_model, 'src/models/best_absenteeism_model.pkl')
    print(f"\nBest model ({best_model_name}) saved with accuracy: {best_accuracy:.2f}")


if __name__ == "__main__":
    main()
