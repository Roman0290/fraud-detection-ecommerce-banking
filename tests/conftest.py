"""
Shared fixtures: small synthetic datasets with the same schema as the real ones.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, 'data')


def make_fraud_df(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Synthetic Fraud_Data.csv: fraud rows buy within seconds of signup on shared devices."""
    rng = np.random.default_rng(seed)
    fraud = rng.random(n) < 0.15
    signup = pd.Timestamp('2015-01-01') + pd.to_timedelta(rng.integers(0, 200 * 86400, n), unit='s')
    delay = np.where(fraud, rng.integers(1, 5, n), rng.integers(3600, 90 * 86400, n))
    devices = np.array([f'DEV{i:05d}' for i in range(n)])
    return pd.DataFrame({
        'user_id': rng.permutation(n) + 1,
        'signup_time': signup,
        'purchase_time': signup + pd.to_timedelta(delay, unit='s'),
        'purchase_value': rng.integers(9, 150, n),
        'device_id': np.where(fraud, rng.choice(devices[:10], n), rng.choice(devices, n)),
        'source': rng.choice(['SEO', 'Ads', 'Direct'], n),
        'browser': rng.choice(['Chrome', 'Safari', 'IE', 'FireFox', 'Opera'], n),
        'sex': rng.choice(['M', 'F'], n),
        'age': rng.integers(18, 70, n),
        'ip_address': rng.uniform(0, 4e9, n),
        'class': fraud.astype(int),
    })


def make_ip_country_df() -> pd.DataFrame:
    """Synthetic IpAddress_to_Country.csv: 1e8-wide ranges with 2e7 gaps between them."""
    lower = np.arange(0, 4e9, 1.2e8)
    countries = ['United States', 'China', 'Japan', 'Germany', 'Brazil', 'Ethiopia']
    return pd.DataFrame({
        'lower_bound_ip_address': lower.astype(float),
        'upper_bound_ip_address': (lower + 1e8).astype(np.int64),
        'country': [countries[i % len(countries)] for i in range(len(lower))],
    })


@pytest.fixture
def raw_fraud_df() -> pd.DataFrame:
    return make_fraud_df()


@pytest.fixture
def ip_country_df() -> pd.DataFrame:
    return make_ip_country_df()


@pytest.fixture
def prepared_fraud(raw_fraud_df, ip_country_df):
    """(X, y) from prepare_fraud_data on the synthetic data."""
    from scripts.features.engineering import prepare_fraud_data
    return prepare_fraud_data(raw_fraud_df, ip_country_df=ip_country_df)


@pytest.fixture
def production_bundle(prepared_fraud, tmp_path):
    """
    Train a small model the way Model_Building does and save it as a bundle.

    Returns (bundle_path, X_test_raw) where X_test_raw holds unencoded feature rows.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from scripts.features.engineering import FeatureEngineer
    from src.models.predict import ModelRegistry

    X, y = prepared_fraud
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    engineer = FeatureEngineer()
    X_train_prep, _ = engineer.prepare_features(X_train, X_test)
    model = RandomForestClassifier(n_estimators=20, random_state=42).fit(X_train_prep, y_train)

    path = ModelRegistry(models_dir=str(tmp_path)).save_model(
        model, 'production_model.pkl', transformer=engineer,
        metadata={'input_features': list(X_train.columns)},
    )
    return path, X_test
