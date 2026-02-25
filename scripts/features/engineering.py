"""
Feature engineering and preprocessing utilities for fraud detection.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handle feature engineering and preprocessing."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = None
        
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input dataframe
            strategy: Imputation strategy ('median', 'mean', 'most_frequent')
            
        Returns:
            DataFrame with imputed values
        """
        df_copy = df.copy()
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        
        if df_copy[numeric_cols].isnull().sum().sum() > 0:
            logger.info(f"Imputing missing values using {strategy} strategy")
            self.imputer = SimpleImputer(strategy=strategy)
            df_copy[numeric_cols] = self.imputer.fit_transform(df_copy[numeric_cols])
      
        categorical_cols = df_copy.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_copy[col].isnull().sum() > 0:
                df_copy[col].fillna(df_copy[col].mode()[0] if len(df_copy[col].mode()) > 0 else 'Unknown', 
                                   inplace=True)
        
        return df_copy
    
    def encode_categorical_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None,
                                   encoding_type: str = 'auto') -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Encode categorical features.
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            encoding_type: 'label', 'onehot', or 'auto' (decides based on cardinality)
            
        Returns:
            Tuple of (X_train_encoded, X_test_encoded)
        """
        X_train_copy = X_train.copy()
        X_test_copy = X_test.copy() if X_test is not None else None
        
        categorical_cols = X_train_copy.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) == 0:
            logger.info("No categorical columns found")
            return X_train_copy, X_test_copy
        
        logger.info(f"Encoding {len(categorical_cols)} categorical columns: {list(categorical_cols)}")
        
        for col in categorical_cols:
            n_unique = X_train_copy[col].nunique()
           
            if encoding_type == 'auto':
                use_onehot = n_unique > 2 and n_unique <= 10
            elif encoding_type == 'onehot':
                use_onehot = True
            else:
                use_onehot = False
            
            if use_onehot:
                
                logger.info(f"  - {col}: One-hot encoding ({n_unique} unique values)")
                X_train_copy = pd.get_dummies(X_train_copy, columns=[col], drop_first=True, prefix=col)
                if X_test_copy is not None:
                    X_test_copy = pd.get_dummies(X_test_copy, columns=[col], drop_first=True, prefix=col)
                    
                    missing_cols = set(X_train_copy.columns) - set(X_test_copy.columns)
                    for c in missing_cols:
                        X_test_copy[c] = 0
                    X_test_copy = X_test_copy[X_train_copy.columns]
            else:
               
                logger.info(f"  - {col}: Label encoding ({n_unique} unique values)")
                le = LabelEncoder()
                X_train_copy[col] = le.fit_transform(X_train_copy[col].astype(str))
                self.label_encoders[col] = le
                
                if X_test_copy is not None:
                    
                    X_test_copy[col] = X_test_copy[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
        
        return X_train_copy, X_test_copy
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            
        Returns:
            Tuple of (X_train_scaled, X_test_scaled)
        """
        logger.info("Scaling numerical features")
        
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        
        return X_train_scaled, X_test_scaled
    
    def create_time_features(self, df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """
        Create time-based features from datetime column.
        
        Args:
            df: Input dataframe
            time_col: Name of datetime column
            
        Returns:
            DataFrame with additional time features
        """
        df_copy = df.copy()
        
        if time_col not in df_copy.columns:
            logger.warning(f"Time column '{time_col}' not found")
            return df_copy
        
        logger.info(f"Creating time features from '{time_col}'")
        
        if not pd.api.types.is_datetime64_any_dtype(df_copy[time_col]):
            df_copy[time_col] = pd.to_datetime(df_copy[time_col])
        
        df_copy['hour'] = df_copy[time_col].dt.hour
        df_copy['day_of_week'] = df_copy[time_col].dt.dayofweek
        df_copy['day_of_month'] = df_copy[time_col].dt.day
        df_copy['month'] = df_copy[time_col].dt.month
        df_copy['is_weekend'] = (df_copy['day_of_week'] >= 5).astype(int)
        
        df_copy['time_of_day'] = pd.cut(df_copy['hour'], 
                                        bins=[0, 6, 12, 18, 24], 
                                        labels=['night', 'morning', 'afternoon', 'evening'],
                                        include_lowest=True)
        
        logger.info(f"Created time features: hour, day_of_week, day_of_month, month, is_weekend, time_of_day")
        
        return df_copy
    
    def create_transaction_velocity_features(self, df: pd.DataFrame, 
                                            user_col: str, 
                                            time_col: str,
                                            amount_col: Optional[str] = None) -> pd.DataFrame:
        """
        Create transaction velocity features (transaction frequency, amount patterns).
        
        Args:
            df: Input dataframe
            user_col: Column identifying users
            time_col: Datetime column
            amount_col: Transaction amount column (optional)
            
        Returns:
            DataFrame with velocity features
        """
        df_copy = df.copy()
        
        if user_col not in df_copy.columns or time_col not in df_copy.columns:
            logger.warning(f"Required columns not found: {user_col}, {time_col}")
            return df_copy
        
        logger.info("Creating transaction velocity features")
        
       
        df_copy = df_copy.sort_values([user_col, time_col])
        
        df_copy['user_transaction_count'] = df_copy.groupby(user_col).cumcount() + 1
        
        df_copy['time_since_last_transaction'] = df_copy.groupby(user_col)[time_col].diff().dt.total_seconds()
        df_copy['time_since_last_transaction'].fillna(0, inplace=True)
        
      
        df_copy['transaction_frequency'] = df_copy.groupby(user_col)[user_col].transform('count')
        
        if amount_col and amount_col in df_copy.columns:
            
            df_copy['user_avg_amount'] = df_copy.groupby(user_col)[amount_col].transform('mean')
            df_copy['user_std_amount'] = df_copy.groupby(user_col)[amount_col].transform('std')
            df_copy['user_max_amount'] = df_copy.groupby(user_col)[amount_col].transform('max')
            
            
            df_copy['amount_deviation'] = (df_copy[amount_col] - df_copy['user_avg_amount']) / (df_copy['user_std_amount'] + 1e-6)
        
        logger.info("Created velocity features: user_transaction_count, time_since_last_transaction, transaction_frequency")
        
        return df_copy
    
    def create_geolocation_features(self, df: pd.DataFrame, ip_country_df: pd.DataFrame,
                                   ip_col: str = 'ip_address') -> pd.DataFrame:
        """
        Merge IP geolocation data and create location-based features.
        
        Args:
            df: Main dataframe
            ip_country_df: IP-to-country mapping dataframe
            ip_col: Column name for IP address
            
        Returns:
            DataFrame with geolocation features
        """
        df_copy = df.copy()
        
        if ip_col not in df_copy.columns:
            logger.warning(f"IP column '{ip_col}' not found")
            return df_copy
        
        logger.info("Creating geolocation features")
        
        df_copy = df_copy.merge(ip_country_df, left_on=ip_col, right_on='lower_bound_ip_address', how='left')
        
        if 'class' in df_copy.columns:
            country_fraud_rate = df_copy.groupby('country')['class'].mean()
            df_copy['country_fraud_rate'] = df_copy['country'].map(country_fraud_rate)
        
        logger.info("Created geolocation features")
        
        return df_copy
    
    def prepare_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None,
                        scale: bool = True, 
                        encode: bool = True) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Complete feature preparation pipeline.
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            scale: Whether to scale features
            encode: Whether to encode categorical features
            
        Returns:
            Tuple of (X_train_prepared, X_test_prepared)
        """
        logger.info("Starting feature preparation pipeline")
        
    
        X_train = self.handle_missing_values(X_train)
        if X_test is not None:
            X_test = self.handle_missing_values(X_test)
        
       
        if encode:
            X_train, X_test = self.encode_categorical_features(X_train, X_test)
        
        
        if scale:
            X_train, X_test = self.scale_features(X_train, X_test)
        
        self.feature_names = X_train.columns.tolist()
        logger.info(f"Feature preparation complete. Final shape: {X_train.shape}")
        
        return X_train, X_test


def prepare_fraud_data(fraud_df: pd.DataFrame, 
                       target_col: str = 'class',
                       ip_country_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare fraud dataset for modeling.
    
    Args:
        fraud_df: Fraud dataframe
        target_col: Name of target column
        ip_country_df: Optional IP-to-country mapping
        
    Returns:
        Tuple of (X, y)
    """
    df = fraud_df.copy()
    
    
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    return X, y


def prepare_credit_data(credit_df: pd.DataFrame, 
                       target_col: str = 'Class') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare credit card dataset for modeling.
    
    Args:
        credit_df: Credit card dataframe
        target_col: Name of target column
        
    Returns:
        Tuple of (X, y)
    """
    df = credit_df.copy()
    

    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    return X, y


if __name__ == "__main__":
   
    from scripts.data.loader import DataLoader
    
    loader = DataLoader()
    fraud_data = loader.load_fraud_data(validate=False)
    
    X, y = prepare_fraud_data(fraud_data)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    engineer = FeatureEngineer()
    X_train_prep, X_test_prep = engineer.prepare_features(X_train, X_test)
    
    print(f"Prepared training set shape: {X_train_prep.shape}")
    print(f"Prepared test set shape: {X_test_prep.shape}")
