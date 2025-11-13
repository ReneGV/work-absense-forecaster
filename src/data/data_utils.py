"""
Data utilities for loading, preparing, and saving data.
"""
import pandas as pd


def load_and_prepare_data(filepath: str, normalize_columns: bool = True) -> pd.DataFrame:
    """
    Load data from CSV and prepare it for prediction.
    
    Args:
        filepath: Path to the CSV file
        normalize_columns: Whether to normalize column names
        
    Returns:
        DataFrame with loaded and prepared data
    """
    df = pd.read_csv(filepath)
    if normalize_columns:
        df.columns = df.columns.str.lower().str.replace("[ ]", "_", regex=True)
    return df

