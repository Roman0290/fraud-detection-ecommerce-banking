"""
Tests for data loading and validation functionality.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.data.loader import DataLoader, DataValidator, load_datasets


class TestDataValidator:
    """Test DataValidator class."""
    
    def test_check_missing_values(self):
        """Test missing value detection."""
        df = pd.DataFrame({
            'a': [1, 2, None, 4],
            'b': [1, 2, 3, 4],
            'c': [None, None, 3, 4]
        })
        
        validator = DataValidator()
        info = validator.check_missing_values(df, "test")
        
        assert info['total_missing'] == 3
        assert 'a' in info['columns_with_missing']
        assert 'c' in info['columns_with_missing']
        assert 'b' not in info['columns_with_missing']
    
    def test_check_duplicates(self):
        """Test duplicate detection."""
        df = pd.DataFrame({
            'a': [1, 2, 2, 3],
            'b': [1, 2, 2, 3]
        })
        
        validator = DataValidator()
        n_duplicates = validator.check_duplicates(df, "test")
        
        assert n_duplicates == 1
    
    def test_check_class_imbalance(self):
        """Test class imbalance detection."""
        df = pd.DataFrame({
            'class': [0, 0, 0, 0, 1]
        })
        
        validator = DataValidator()
        info = validator.check_class_imbalance(df, 'class', "test")
        
        assert info['imbalance_ratio'] == 4.0
        assert info['counts'][0] == 4
        assert info['counts'][1] == 1


class TestDataLoader:
    """Test DataLoader class."""
    
    @pytest.fixture
    def data_dir(self, tmp_path):
        """Create temporary data directory with test files."""
        # Create test CSV files
        fraud_data = pd.DataFrame({
            'user_id': [1, 2, 3],
            'purchase_value': [100, 200, 50],
            'class': [0, 1, 0]
        })
        
        credit_data = pd.DataFrame({
            'V1': [0.1, 0.2, 0.3],
            'V2': [1.1, 1.2, 1.3],
            'Class': [0, 0, 1]
        })
        
        fraud_data.to_csv(tmp_path / 'cleaned_data_1.csv', index=False)
        credit_data.to_csv(tmp_path / 'cleaned_data_2.csv', index=False)
        
        return tmp_path
    
    def test_load_fraud_data(self, data_dir):
        """Test loading fraud data."""
        loader = DataLoader(data_dir=str(data_dir))
        df = loader.load_fraud_data(validate=False)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'class' in df.columns
    
    def test_load_credit_data(self, data_dir):
        """Test loading credit data."""
        loader = DataLoader(data_dir=str(data_dir))
        df = loader.load_credit_data(validate=False)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'Class' in df.columns
    
    def test_load_both_datasets(self, data_dir):
        """Test loading both datasets."""
        loader = DataLoader(data_dir=str(data_dir))
        fraud_df, credit_df = loader.load_both_datasets(validate=False)
        
        assert isinstance(fraud_df, pd.DataFrame)
        assert isinstance(credit_df, pd.DataFrame)
        assert len(fraud_df) == 3
        assert len(credit_df) == 3


def test_load_datasets_function(tmp_path):
    """Test convenience function for loading datasets."""
    # Create test data
    fraud_data = pd.DataFrame({
        'user_id': [1, 2, 3],
        'class': [0, 1, 0]
    })
    credit_data = pd.DataFrame({
        'V1': [0.1, 0.2, 0.3],
        'Class': [0, 0, 1]
    })
    
    fraud_data.to_csv(tmp_path / 'cleaned_data_1.csv', index=False)
    credit_data.to_csv(tmp_path / 'cleaned_data_2.csv', index=False)
    
    datasets = load_datasets(data_dir=str(tmp_path), validate=False)
    
    assert 'fraud' in datasets
    assert 'credit' in datasets
    assert isinstance(datasets['fraud'], pd.DataFrame)
    assert isinstance(datasets['credit'], pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
