"""
Tests for preprocessor transformers in src/models/preprocessors.py
"""
import pytest
import pandas as pd
import numpy as np
from src.models.preprocessors import (
    DropColumnsTransformer,
    IQRClippingTransformer,
    ToStringTransformer
)


class TestDropColumnsTransformer:
    """Tests for DropColumnsTransformer"""

    def test_drop_columns_transformer_init(self):
        """Test initialization of DropColumnsTransformer"""
        columns = ['col1', 'col2']
        transformer = DropColumnsTransformer(columns)
        assert transformer.columns_to_drop == columns

    def test_drop_columns_transform(self, sample_dataframe, columns_to_drop):
        """Test that specified columns are dropped"""
        transformer = DropColumnsTransformer(columns_to_drop)
        transformer.fit(sample_dataframe)
        result = transformer.transform(sample_dataframe)
        
        # Check that dropped columns are not in result
        for col in columns_to_drop:
            assert col not in result.columns
        
        # Check that remaining columns are present
        remaining_cols = [col for col in sample_dataframe.columns if col not in columns_to_drop]
        for col in remaining_cols:
            assert col in result.columns

    def test_drop_nonexistent_columns(self, sample_dataframe):
        """Test dropping columns that don't exist in the DataFrame"""
        transformer = DropColumnsTransformer(['nonexistent_col1', 'nonexistent_col2'])
        transformer.fit(sample_dataframe)
        result = transformer.transform(sample_dataframe)
        
        # DataFrame should remain unchanged
        assert result.shape == sample_dataframe.shape

    def test_drop_partial_columns(self, sample_dataframe):
        """Test dropping mix of existing and non-existing columns"""
        columns = ['id', 'nonexistent_col']
        transformer = DropColumnsTransformer(columns)
        transformer.fit(sample_dataframe)
        result = transformer.transform(sample_dataframe)
        
        # Only 'id' should be dropped
        assert 'id' not in result.columns
        assert result.shape[1] == sample_dataframe.shape[1] - 1

    def test_get_feature_names_out(self, sample_dataframe, columns_to_drop):
        """Test get_feature_names_out method"""
        transformer = DropColumnsTransformer(columns_to_drop)
        input_features = sample_dataframe.columns.tolist()
        output_features = transformer.get_feature_names_out(input_features)
        
        # Check that dropped columns are not in output
        for col in columns_to_drop:
            assert col not in output_features
        
        # Check expected number of features
        expected_count = len(input_features) - len(columns_to_drop)
        assert len(output_features) == expected_count

    def test_get_feature_names_out_no_input(self):
        """Test get_feature_names_out raises error when no input provided"""
        transformer = DropColumnsTransformer(['col1'])
        with pytest.raises(ValueError, match="input_features must be provided"):
            transformer.get_feature_names_out()


class TestIQRClippingTransformer:
    """Tests for IQRClippingTransformer"""

    def test_iqr_clipping_transformer_init(self):
        """Test initialization of IQRClippingTransformer"""
        transformer = IQRClippingTransformer()
        assert transformer.lower_bounds_ == {}
        assert transformer.upper_bounds_ == {}

    def test_iqr_clipping_fit(self, numerical_dataframe):
        """Test that IQR bounds are calculated correctly"""
        transformer = IQRClippingTransformer()
        transformer.fit(numerical_dataframe)
        
        # Check that bounds are calculated for all columns
        assert len(transformer.lower_bounds_) == numerical_dataframe.shape[1]
        assert len(transformer.upper_bounds_) == numerical_dataframe.shape[1]
        
        # Check that bounds exist for each column
        for col in numerical_dataframe.columns:
            assert col in transformer.lower_bounds_
            assert col in transformer.upper_bounds_

    def test_iqr_clipping_transform_outliers(self):
        """Test that outliers are clipped correctly"""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5, 100],  # 100 is an outlier
        })
        
        transformer = IQRClippingTransformer()
        transformer.fit(df)
        result = transformer.transform(df)
        
        # Check that the outlier is clipped
        assert result['col1'].max() < 100
        assert result['col1'].min() >= df['col1'].min()

    def test_iqr_clipping_no_outliers(self):
        """Test that normal values are not affected"""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
        })
        
        transformer = IQRClippingTransformer()
        transformer.fit(df)
        result = transformer.transform(df)
        
        # Values should remain mostly unchanged (within IQR bounds)
        pd.testing.assert_frame_equal(result, df, check_dtype=False)

    def test_iqr_clipping_transform_preserves_shape(self, numerical_dataframe):
        """Test that transform preserves DataFrame shape"""
        transformer = IQRClippingTransformer()
        transformer.fit(numerical_dataframe)
        result = transformer.transform(numerical_dataframe)
        
        assert result.shape == numerical_dataframe.shape

    def test_get_feature_names_out(self):
        """Test get_feature_names_out method"""
        transformer = IQRClippingTransformer()
        input_features = ['col1', 'col2', 'col3']
        output_features = transformer.get_feature_names_out(input_features)
        
        assert output_features == input_features

    def test_get_feature_names_out_none(self):
        """Test get_feature_names_out with None input"""
        transformer = IQRClippingTransformer()
        output_features = transformer.get_feature_names_out(None)
        assert output_features == []


class TestToStringTransformer:
    """Tests for ToStringTransformer"""

    def test_to_string_transformer_init(self):
        """Test initialization of ToStringTransformer"""
        transformer = ToStringTransformer()
        assert transformer is not None

    def test_to_string_transform_numeric(self):
        """Test converting numeric columns to strings"""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [10, 20, 30]
        })
        
        transformer = ToStringTransformer()
        transformer.fit(df)
        result = transformer.transform(df)
        
        # Check that all columns are strings
        for col in result.columns:
            assert result[col].dtype == 'object'
            assert all(isinstance(val, str) for val in result[col])

    def test_to_string_transform_mixed_types(self):
        """Test converting mixed type columns to strings"""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c'],
            'col3': [1.5, 2.5, 3.5]
        })
        
        transformer = ToStringTransformer()
        transformer.fit(df)
        result = transformer.transform(df)
        
        # Check that all columns are strings
        for col in result.columns:
            assert result[col].dtype == 'object'
            assert all(isinstance(val, str) for val in result[col])

    def test_to_string_transform_preserves_shape(self, categorical_dataframe):
        """Test that transform preserves DataFrame shape"""
        transformer = ToStringTransformer()
        transformer.fit(categorical_dataframe)
        result = transformer.transform(categorical_dataframe)
        
        assert result.shape == categorical_dataframe.shape

    def test_to_string_values_correct(self):
        """Test that string conversion produces correct values"""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
        })
        
        transformer = ToStringTransformer()
        transformer.fit(df)
        result = transformer.transform(df)
        
        assert result['col1'].tolist() == ['1', '2', '3']

    def test_get_feature_names_out(self):
        """Test get_feature_names_out method"""
        transformer = ToStringTransformer()
        input_features = ['col1', 'col2', 'col3']
        output_features = transformer.get_feature_names_out(input_features)
        
        assert output_features == input_features

    def test_get_feature_names_out_none(self):
        """Test get_feature_names_out with None input"""
        transformer = ToStringTransformer()
        output_features = transformer.get_feature_names_out(None)
        assert output_features == []


class TestTransformerIntegration:
    """Integration tests for using transformers together"""

    def test_pipeline_with_all_transformers(self, sample_dataframe, columns_to_drop):
        """Test using all transformers in sequence"""
        from sklearn.pipeline import Pipeline
        
        # Create pipeline
        pipeline = Pipeline([
            ('drop', DropColumnsTransformer(columns_to_drop)),
            ('to_string', ToStringTransformer())
        ])
        
        # Fit and transform
        pipeline.fit(sample_dataframe)
        result = pipeline.transform(sample_dataframe)
        
        # Verify columns are dropped
        for col in columns_to_drop:
            assert col not in result.columns
        
        # Verify all remaining columns are strings
        for col in result.columns:
            assert result[col].dtype == 'object'

    def test_transformers_with_sklearn_compatibility(self, numerical_dataframe):
        """Test that transformers work with sklearn's ColumnTransformer"""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler
        
        # Create a composite transformer
        ct = ColumnTransformer([
            ('iqr', IQRClippingTransformer(), ['col1', 'col2']),
            ('scaler', StandardScaler(), ['col3'])
        ])
        
        # Should fit without errors
        ct.fit(numerical_dataframe)
        result = ct.transform(numerical_dataframe)
        
        # Check output shape
        assert result.shape[0] == numerical_dataframe.shape[0]

