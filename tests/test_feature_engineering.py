"""
Tests for feature engineering functionality.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.features.engineering import FeatureEngineer, prepare_fraud_data, prepare_credit_data


class TestFeatureEngineer:
    """Test FeatureEngineer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5],
            'categorical_col': ['A', 'B', 'A', 'C', 'B'],
            'binary_col': ['yes', 'no', 'yes', 'no', 'yes']
        })
    
    def test_handle_missing_values(self, sample_data):
        """Test missing value imputation."""
        df = sample_data.copy()
        df.loc[0, 'numeric_col'] = np.nan
        
        engineer = FeatureEngineer()
        df_imputed = engineer.handle_missing_values(df, strategy='median')
        
        assert not df_imputed['numeric_col'].isnull().any()
    
    def test_encode_categorical_features(self, sample_data):
        """Test categorical encoding."""
        train_df = sample_data.copy()
        test_df = sample_data.copy()
        
        engineer = FeatureEngineer()
        train_encoded, test_encoded = engineer.encode_categorical_features(
            train_df, test_df, encoding_type='auto'
        )
        
        # Check that categorical columns are encoded
        assert train_encoded.select_dtypes(include=['object']).shape[1] == 0
        assert test_encoded.select_dtypes(include=['object']).shape[1] == 0
    
    def test_scale_features(self, sample_data):
        """Test feature scaling."""
        numeric_df = sample_data[['numeric_col']].copy()
        train_df = numeric_df.iloc[:3]
        test_df = numeric_df.iloc[3:]
        
        engineer = FeatureEngineer()
        train_scaled, test_scaled = engineer.scale_features(train_df, test_df)
        
        # Check that mean is close to 0 and std is close to 1
        assert abs(train_scaled['numeric_col'].mean()) < 0.1
        assert abs(train_scaled['numeric_col'].std() - 1.0) < 0.1
    
    def test_create_time_features(self):
        """Test time feature creation."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='H')
        })
        
        engineer = FeatureEngineer()
        df_with_time = engineer.create_time_features(df, 'timestamp')
        
        assert 'hour' in df_with_time.columns
        assert 'day_of_week' in df_with_time.columns
        assert 'is_weekend' in df_with_time.columns
        assert 'time_of_day' in df_with_time.columns
    
    def test_prepare_features(self, sample_data):
        """Test complete feature preparation pipeline."""
        train_df = sample_data.copy()
        test_df = sample_data.copy()
        
        engineer = FeatureEngineer()
        train_prep, test_prep = engineer.prepare_features(train_df, test_df)
        
        # Check that output is numeric
        assert train_prep.select_dtypes(include=['object']).shape[1] == 0
        assert test_prep.select_dtypes(include=['object']).shape[1] == 0
        
        # Check shapes match
        assert train_prep.shape[1] == test_prep.shape[1]


def test_prepare_fraud_data():
    """Test fraud data preparation."""
    fraud_df = pd.DataFrame({
        'user_id': [1, 2, 3],
        'purchase_value': [100, 200, 50],
        'class': [0, 1, 0]
    })
    
    X, y = prepare_fraud_data(fraud_df, target_col='class')
    
    assert 'class' not in X.columns
    assert len(y) == len(fraud_df)
    assert list(y) == [0, 1, 0]


def test_prepare_credit_data():
    """Test credit data preparation."""
    credit_df = pd.DataFrame({
        'V1': [0.1, 0.2, 0.3],
        'V2': [1.1, 1.2, 1.3],
        'Class': [0, 0, 1]
    })
    
    X, y = prepare_credit_data(credit_df, target_col='Class')
    
    assert 'Class' not in X.columns
    assert len(y) == len(credit_df)
    assert list(y) == [0, 0, 1]


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
