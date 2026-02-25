"""
Data loading and validation utilities for fraud detection datasets.
"""
import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data integrity and quality."""
    
    @staticmethod
    def check_missing_values(df: pd.DataFrame, dataset_name: str) -> Dict:
        """Check for missing values in the dataset."""
        missing_info = {}
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df)) * 100
        
        missing_info['total_missing'] = missing_counts.sum()
        missing_info['columns_with_missing'] = missing_counts[missing_counts > 0].to_dict()
        missing_info['missing_percentages'] = missing_pct[missing_pct > 0].to_dict()
        
        if missing_info['total_missing'] > 0:
            logger.warning(f"{dataset_name}: Found {missing_info['total_missing']} missing values")
            for col, count in missing_info['columns_with_missing'].items():
                logger.warning(f"  - {col}: {count} ({missing_pct[col]:.2f}%)")
        else:
            logger.info(f"{dataset_name}: No missing values found")
        
        return missing_info
    
    @staticmethod
    def check_duplicates(df: pd.DataFrame, dataset_name: str) -> int:
        """Check for duplicate rows."""
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            logger.warning(f"{dataset_name}: Found {n_duplicates} duplicate rows ({n_duplicates/len(df)*100:.2f}%)")
        else:
            logger.info(f"{dataset_name}: No duplicates found")
        return n_duplicates
    
    @staticmethod
    def check_data_types(df: pd.DataFrame, dataset_name: str) -> Dict:
        """Check data types of columns."""
        dtype_info = df.dtypes.value_counts().to_dict()
        logger.info(f"{dataset_name}: Data types - {dtype_info}")
        return dtype_info
    
    @staticmethod
    def check_class_imbalance(df: pd.DataFrame, target_col: str, dataset_name: str) -> Dict:
        """Check class distribution for imbalance."""
        class_counts = df[target_col].value_counts()
        class_pcts = (class_counts / len(df) * 100).round(2)
        
        imbalance_info = {
            'counts': class_counts.to_dict(),
            'percentages': class_pcts.to_dict(),
            'imbalance_ratio': class_counts.max() / class_counts.min()
        }
        
        logger.info(f"{dataset_name}: Class distribution:")
        for cls, count in class_counts.items():
            logger.info(f"  - Class {cls}: {count} ({class_pcts[cls]}%)")
        logger.info(f"  - Imbalance ratio: {imbalance_info['imbalance_ratio']:.2f}:1")
        
        return imbalance_info
    
    @staticmethod
    def check_outliers(df: pd.DataFrame, dataset_name: str, n_std: float = 3.0) -> Dict:
        """Check for outliers using standard deviation method."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            outliers = df[(df[col] < mean - n_std * std) | (df[col] > mean + n_std * std)]
            if len(outliers) > 0:
                outlier_info[col] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(df) * 100
                }
        
        if outlier_info:
            logger.warning(f"{dataset_name}: Found outliers in {len(outlier_info)} columns")
            for col, info in outlier_info.items():
                logger.warning(f"  - {col}: {info['count']} outliers ({info['percentage']:.2f}%)")
        else:
            logger.info(f"{dataset_name}: No significant outliers detected")
        
        return outlier_info


class DataLoader:
    """Loads and validates fraud detection datasets."""
    
    def __init__(self, data_dir: str = None):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Path to data directory. If None, uses default '../data' relative to this file.
        """
        if data_dir is None:
           
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_dir = os.path.join(script_dir, '..', '..', 'data')
        else:
            self.data_dir = data_dir
        
        self.validator = DataValidator()
        logger.info(f"DataLoader initialized with data directory: {self.data_dir}")
    
    def load_fraud_data(self, validate: bool = True) -> pd.DataFrame:
        """
        Load e-commerce fraud dataset (Fraud_Data.csv or fraud_data_combined.csv).
        
        Args:
            validate: Whether to run validation checks
            
        Returns:
            DataFrame containing fraud data
        """
        
        cleaned_path = os.path.join(self.data_dir, 'fraud_data_combined.csv')
        raw_path = os.path.join(self.data_dir, 'Fraud_Data.csv')
        
        if os.path.exists(cleaned_path):
            logger.info(f"Loading cleaned fraud data from {cleaned_path}")
            df = pd.read_csv(cleaned_path)
        elif os.path.exists(raw_path):
            logger.info(f"Loading raw fraud data from {raw_path}")
            df = pd.read_csv(raw_path)
        else:
            raise FileNotFoundError(f"Fraud data not found in {self.data_dir}")
        
        logger.info(f"Loaded fraud dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        
        if validate:
            self._validate_dataset(df, "Fraud Data", target_col='class')
        
        return df
    
    def load_credit_data(self, validate: bool = True) -> pd.DataFrame:
        """
        Load credit card fraud dataset (creditcard.csv or credit_cleaned_data_2.csv).
        
        Args:
            validate: Whether to run validation checks
            
        Returns:
            DataFrame containing credit data
        """
        # Try cleaned data first, fall back to raw
        cleaned_path = os.path.join(self.data_dir, 'credit_cleaned_data_2.csv')
        raw_path = os.path.join(self.data_dir, 'creditcard.csv')
        
        if os.path.exists(cleaned_path):
            logger.info(f"Loading cleaned credit data from {cleaned_path}")
            df = pd.read_csv(cleaned_path)
        elif os.path.exists(raw_path):
            logger.info(f"Loading raw credit data from {raw_path}")
            df = pd.read_csv(raw_path)
        else:
            raise FileNotFoundError(f"Credit data not found in {self.data_dir}")
        
        logger.info(f"Loaded credit dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        
        if validate:
            self._validate_dataset(df, "Credit Data", target_col='Class')
        
        return df
    
    def load_ip_country_data(self) -> pd.DataFrame:
        """
        Load IP address to country mapping dataset.
        
        Returns:
            DataFrame containing IP-to-country mappings
        """
        path = os.path.join(self.data_dir, 'IpAddress_to_Country.csv')
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"IP country data not found at {path}")
        
        logger.info(f"Loading IP-to-country data from {path}")
        df = pd.read_csv(path)
        logger.info(f"Loaded IP country dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return df
    
    def load_both_datasets(self, validate: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both fraud and credit datasets.
        
        Args:
            validate: Whether to run validation checks
            
        Returns:
            Tuple of (fraud_data, credit_data)
        """
        fraud_data = self.load_fraud_data(validate=validate)
        credit_data = self.load_credit_data(validate=validate)
        
        return fraud_data, credit_data
    
    def _validate_dataset(self, df: pd.DataFrame, dataset_name: str, target_col: str):
        """Run all validation checks on a dataset."""
        logger.info(f"\n{'='*50}")
        logger.info(f"Validating {dataset_name}")
        logger.info(f"{'='*50}")
        
        self.validator.check_missing_values(df, dataset_name)
        
        self.validator.check_duplicates(df, dataset_name)
        
        self.validator.check_data_types(df, dataset_name)
        
        if target_col in df.columns:
            self.validator.check_class_imbalance(df, target_col, dataset_name)
        
        self.validator.check_outliers(df, dataset_name)
        
        logger.info(f"{'='*50}\n")


def prepare_train_test_split(X: pd.DataFrame, y: pd.Series, 
                             test_size: float = 0.2, 
                             random_state: int = 42) -> Tuple:
    """
    Prepare train-test split with stratification.
    
    Args:
        X: Features
        y: Target variable
        test_size: Proportion of test set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    return train_test_split(X, y, test_size=test_size, 
                          random_state=random_state, 
                          stratify=y)


def load_datasets(data_dir: str = None, validate: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Load all datasets and return as dictionary.
    
    Args:
        data_dir: Path to data directory
        validate: Whether to run validation checks
        
    Returns:
        Dictionary with keys: 'fraud', 'credit', 'ip_country'
    """
    loader = DataLoader(data_dir)
    
    datasets = {
        'fraud': loader.load_fraud_data(validate=validate),
        'credit': loader.load_credit_data(validate=validate)
    }
    
    try:
        datasets['ip_country'] = loader.load_ip_country_data()
    except FileNotFoundError:
        logger.warning("IP country data not found, skipping...")
    
    return datasets


if __name__ == "__main__":
    loader = DataLoader()
    fraud_data, credit_data = loader.load_both_datasets(validate=True)
    print(f"\nFraud data shape: {fraud_data.shape}")
    print(f"Credit data shape: {credit_data.shape}")
