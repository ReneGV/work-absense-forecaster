import pandas as pd
import joblib
import mlflow
from src.models.preprocessors import DropColumnsTransformer, IQRClippingTransformer, ToStringTransformer
from src.data.data_utils import load_and_prepare_data
from src.models.evaluation import evaluate_predictions, get_classification_report, create_confusion_matrix
from typing import Tuple, Dict, Optional
import numpy as np


def load_model(model_path: str):
    """
    Load a trained model from a pickle file.
    
    Args:
        model_path: Path to the model pickle file
        
    Returns:
        Loaded model object
    """
    return joblib.load(model_path)


def prepare_test_data(df: pd.DataFrame, 
                     sample_size: int = 25, 
                     random_state: int = 42,
                     columns_to_drop: Optional[list] = None) -> pd.DataFrame:
    """
    Prepare test data from original dataset by sampling and preserving ground truth.
    
    Args:
        df: Original dataframe
        sample_size: Number of samples to take
        random_state: Random state for reproducibility
        columns_to_drop: Columns to drop from the dataset (e.g., ['id', 'body_mass_index'])
        
    Returns:
        DataFrame with test data and ground truth column
    """
    if columns_to_drop is None:
        columns_to_drop = ['id', 'body_mass_index']
    
    df_new = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Preserve ground truth if absenteeism_time_in_hours exists
    if 'absenteeism_time_in_hours' in df_new.columns:
        df_new['absenteeism_real'] = df_new['absenteeism_time_in_hours']
        df_new = df_new.drop(columns=['absenteeism_time_in_hours'])
    
    # Sample data
    df_new = df_new.sample(sample_size, random_state=random_state)
    
    return df_new


def make_predictions(model, data: pd.DataFrame, exclude_columns: Optional[list] = None) -> np.ndarray:
    """
    Make predictions using a trained model.
    
    Args:
        model: Trained model with predict method
        data: DataFrame with features for prediction
        exclude_columns: Columns to exclude from prediction (e.g., ground truth)
        
    Returns:
        Array of predictions
    """
    if exclude_columns is None:
        exclude_columns = ['absenteeism_real']
    
    X = data.drop(columns=exclude_columns, errors='ignore')
    predictions = model.predict(X)
    
    return predictions


def convert_to_labels(predictions: np.ndarray) -> list:
    """
    Convert numeric predictions to human-readable labels.
    
    Args:
        predictions: Array of numeric predictions (0 or 1)
        
    Returns:
        List of string labels ('High' or 'Low')
    """
    return ['High' if p == 1 else 'Low' for p in predictions]


def save_predictions(data: pd.DataFrame, output_path: str) -> None:
    """
    Save predictions to a CSV file.
    
    Args:
        data: DataFrame with predictions
        output_path: Path where to save the CSV file
    """
    data.to_csv(output_path, index=False)


def convert_to_binary(values: pd.Series, threshold: Optional[float] = None) -> pd.Series:
    """
    Convert continuous values to binary using a threshold (default: median).
    
    Args:
        values: Series of continuous values
        threshold: Threshold value (if None, uses median)
        
    Returns:
        Series of binary values (0 or 1)
    """
    if threshold is None:
        threshold = values.median()
    
    return (values > threshold).astype(int)


def main():
    """
    Main function to run the prediction pipeline with MLflow tracking.
    """
    # Configure MLflow
    mlflow.set_experiment("absenteeism_predictions")

    with mlflow.start_run():
        # Load dataset and generate test data
        df = load_and_prepare_data('data/raw/work_absenteeism_original.csv')
        
        # Prepare test data
        df_new = prepare_test_data(df, sample_size=25, random_state=42)
        df_new.to_csv('data/raw/new_absenteeism_data.csv', index=False)

        # Load model
        model_path = 'src/models/best_absenteeism_model.pkl'
        model = load_model(model_path)

        # Make predictions
        new_data = load_and_prepare_data('data/raw/new_absenteeism_data.csv', 
                                        normalize_columns=False)
        predictions = make_predictions(model, new_data)
        prediction_labels = convert_to_labels(predictions)
        new_data['predicted_absenteeism'] = prediction_labels

        # Save predictions
        output_path = 'data/predictions/absenteeism_predictions.csv'
        save_predictions(new_data, output_path)
        mlflow.log_artifact(output_path)
        print("\n Predicciones guardadas y logueadas en MLflow")

        # Evaluate if ground truth exists
        if 'absenteeism_real' in new_data.columns:
            y_true = convert_to_binary(new_data['absenteeism_real'])
            
            # Calculate and log metrics
            metrics = evaluate_predictions(y_true, predictions)
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            print(f"\nAccuracy: {metrics['accuracy']:.2f}")
            print("\nClassification Report:")
            print(get_classification_report(y_true, predictions))

            # Create and log confusion matrix
            confusion_matrix_path = "confusion_matrix.png"
            create_confusion_matrix(y_true, predictions, confusion_matrix_path)
            mlflow.log_artifact(confusion_matrix_path)


if __name__ == "__main__":
    main()
