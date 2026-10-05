"""
Data integrity tests: schema, nulls, duplicates and label values.

The pipeline tests run on synthetic data. The TestRealData tests check the files
in data/ and are skipped when they are not present (e.g. in CI).
"""
import os

import numpy as np
import pandas as pd
import pytest

from scripts.data.loader import DataValidator
from scripts.features.engineering import FeatureEngineer, FRAUD_DROP_COLUMNS, prepare_fraud_data

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

FRAUD_RAW_COLUMNS = [
    'user_id', 'signup_time', 'purchase_time', 'purchase_value', 'device_id',
    'source', 'browser', 'sex', 'age', 'ip_address', 'class',
]
FRAUD_MODEL_FEATURES = [
    'purchase_value', 'source', 'browser', 'sex', 'age', 'country',
    'day_of_week', 'hour_of_day', 'time_since_last_transaction',
    'transaction_count', 'signup_purchase_diff',
]
CREDIT_COLUMNS = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
IP_COLUMNS = ['lower_bound_ip_address', 'upper_bound_ip_address', 'country']


# --- prepare_fraud_data output -------------------------------------------------

class TestPreparedFraudData:
    def test_feature_schema(self, prepared_fraud):
        X, _ = prepared_fraud
        assert sorted(X.columns) == sorted(FRAUD_MODEL_FEATURES)

    def test_no_identifiers_or_raw_timestamps(self, prepared_fraud):
        X, _ = prepared_fraud
        assert not set(FRAUD_DROP_COLUMNS) & set(X.columns)
        assert 'class' not in X.columns

    def test_no_nulls(self, prepared_fraud):
        X, y = prepared_fraud
        assert X.isnull().sum().sum() == 0
        assert y.isnull().sum() == 0

    def test_labels_are_binary(self, prepared_fraud):
        _, y = prepared_fraud
        assert set(y.unique()) <= {0, 1}

    def test_rows_stay_aligned(self, raw_fraud_df, prepared_fraud):
        """Velocity features sort internally; the output must keep the input row order."""
        X, y = prepared_fraud
        assert len(X) == len(y) == len(raw_fraud_df)
        assert (X.index == raw_fraud_df.index).all()
        assert (y.values == raw_fraud_df['class'].values).all()
        assert (X['purchase_value'].values == raw_fraud_df['purchase_value'].values).all()

    def test_feature_values(self, raw_fraud_df, prepared_fraud):
        X, _ = prepared_fraud
        assert X['hour_of_day'].between(0, 23).all()
        assert X['day_of_week'].between(0, 6).all()
        assert (X['signup_purchase_diff'] >= 0).all()
        assert (X['transaction_count'] >= 1).all()
        expected_counts = raw_fraud_df['device_id'].map(raw_fraud_df['device_id'].value_counts())
        assert (X['transaction_count'].values == expected_counts.values).all()

    def test_accepts_csv_strings_and_existing_country(self, raw_fraud_df):
        """fraud_data_combined.csv has string timestamps and a country column already."""
        df = raw_fraud_df.astype({'signup_time': str, 'purchase_time': str})
        df['country'] = 'Japan'
        X, _ = prepare_fraud_data(df)
        assert sorted(X.columns) == sorted(FRAUD_MODEL_FEATURES)
        assert (X['country'] == 'Japan').all()


# --- IP to country merge -------------------------------------------------------

class TestIpCountryMerge:
    @pytest.fixture
    def ranges(self):
        return pd.DataFrame({
            'lower_bound_ip_address': [100.0, 200.0, 300.0],
            'upper_bound_ip_address': [150, 250, 350],
            'country': ['A', 'B', 'C'],
        })

    def test_range_lookup(self, ranges):
        df = pd.DataFrame({'ip_address': [100.0, 125.7, 150.0, 175.0, 250.0, 99.0, 999.0, np.nan]})
        result = FeatureEngineer().merge_ip_country_range(df, ranges)
        assert list(result['country']) == ['A', 'A', 'A', 'Unknown', 'B', 'Unknown', 'Unknown', 'Unknown']

    def test_preserves_order_and_length(self, ranges):
        df = pd.DataFrame({'ip_address': [300.0, 100.0, 200.0]}, index=[10, 5, 7])
        result = FeatureEngineer().merge_ip_country_range(df, ranges)
        assert list(result.index) == [10, 5, 7]
        assert list(result['country']) == ['C', 'A', 'B']

    def test_dotted_ip_strings(self):
        assert FeatureEngineer.convert_ip_to_int('1.0.0.1') == 16777217
        assert FeatureEngineer.convert_ip_to_int(732758368.79) == 732758368
        assert np.isnan(FeatureEngineer.convert_ip_to_int(None))


# --- validators ----------------------------------------------------------------

class TestDataValidator:
    def test_detects_nulls(self):
        df = pd.DataFrame({'a': [1, None, 3], 'b': ['x', 'y', None]})
        info = DataValidator.check_missing_values(df, 'test')
        assert info['total_missing'] == 2
        assert set(info['columns_with_missing']) == {'a', 'b'}

    def test_detects_duplicates(self):
        df = pd.DataFrame({'a': [1, 1, 2], 'b': ['x', 'x', 'y']})
        assert DataValidator.check_duplicates(df, 'test') == 1
        assert len(FeatureEngineer().remove_duplicates(df)) == 2


# --- real data files (skipped when data/ is absent) -----------------------------

def _load_first(*names):
    for name in names:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            return pd.read_csv(path)
    pytest.skip(f"none of {names} found in data/")


class TestRealData:
    def test_fraud_data(self):
        df = _load_first('fraud_data_combined.csv', 'Fraud_Data.csv')
        assert set(FRAUD_RAW_COLUMNS) <= set(df.columns)
        assert df[FRAUD_RAW_COLUMNS].isnull().sum().sum() == 0
        assert not df.duplicated().any()
        assert df['user_id'].is_unique
        assert set(df['class'].unique()) == {0, 1}
        assert (df['purchase_value'] > 0).all()
        assert (pd.to_datetime(df['purchase_time']) >= pd.to_datetime(df['signup_time'])).all()

    def test_credit_data(self):
        df = _load_first('credit_cleaned_data_2.csv', 'creditcard.csv')
        assert set(CREDIT_COLUMNS) <= set(df.columns)
        assert df[CREDIT_COLUMNS].isnull().sum().sum() == 0
        assert set(df['Class'].unique()) == {0, 1}
        assert (df['Amount'] >= 0).all()

    def test_ip_country_data(self):
        df = _load_first('IpAddress_to_Country.csv')
        assert set(IP_COLUMNS) <= set(df.columns)
        assert df[IP_COLUMNS].isnull().sum().sum() == 0
        assert (df['upper_bound_ip_address'] >= df['lower_bound_ip_address']).all()
