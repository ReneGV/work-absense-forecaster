"""
Tests for data utilities in src/data/data_utils.py
"""
import pytest
import pandas as pd
import tempfile
import os
from src.data.data_utils import load_and_prepare_data


class TestLoadAndPrepareData:
    """Tests for load_and_prepare_data function"""

    def test_load_and_prepare_data_basic(self):
        """Test basic data loading without normalization"""
        # Create a simple DataFrame
        df = pd.DataFrame({
            'Column1': [1, 2, 3],
            'Column2': [4, 5, 6]
        })
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w') as tmp:
            tmp_path = tmp.name
        
        try:
            df.to_csv(tmp_path, index=False)
            
            # Load data using the function without normalization
            loaded_df = load_and_prepare_data(tmp_path, normalize_columns=False)
            
            assert len(loaded_df) == len(df)
            assert list(loaded_df.columns) == list(df.columns)
            assert loaded_df.equals(df)
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_load_and_prepare_data_with_normalization(self):
        """Test load_and_prepare_data with column normalization"""
        # Create a DataFrame with columns that need normalization
        df = pd.DataFrame({
            'Column Name': [1, 2, 3],
            'Another Column': [4, 5, 6]
        })
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w') as tmp:
            tmp_path = tmp.name
        
        try:
            df.to_csv(tmp_path, index=False)
            
            # Load with normalization
            loaded_df = load_and_prepare_data(tmp_path, normalize_columns=True)
            
            # Check that columns were normalized
            assert 'column_name' in loaded_df.columns
            assert 'another_column' in loaded_df.columns
            assert 'Column Name' not in loaded_df.columns
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_load_and_prepare_data_with_spaces_in_columns(self):
        """Test that spaces in column names are replaced with underscores"""
        df = pd.DataFrame({
            'First Column': [1, 2, 3],
            'Second  Column': [4, 5, 6],  # Multiple spaces
            'ThirdColumn': [7, 8, 9]
        })
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w') as tmp:
            tmp_path = tmp.name
        
        try:
            df.to_csv(tmp_path, index=False)
            
            # Load with normalization
            loaded_df = load_and_prepare_data(tmp_path, normalize_columns=True)
            
            # Check column names
            assert 'first_column' in loaded_df.columns
            assert 'second__column' in loaded_df.columns  # Multiple spaces preserved as multiple underscores
            assert 'thirdcolumn' in loaded_df.columns
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_load_and_prepare_data_preserves_data_values(self):
        """Test that data values are preserved during loading"""
        df = pd.DataFrame({
            'Value A': [1.5, 2.7, 3.2],
            'Value B': ['x', 'y', 'z'],
            'Value C': [True, False, True]
        })
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w') as tmp:
            tmp_path = tmp.name
        
        try:
            df.to_csv(tmp_path, index=False)
            
            # Load with normalization
            loaded_df = load_and_prepare_data(tmp_path, normalize_columns=True)
            
            # Check that values are preserved
            assert loaded_df['value_a'].tolist() == [1.5, 2.7, 3.2]
            assert loaded_df['value_b'].tolist() == ['x', 'y', 'z']
            assert loaded_df['value_c'].tolist() == [True, False, True]
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

