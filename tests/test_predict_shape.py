"""
Prediction output tests: shapes, probability range and threshold behaviour,
using a model trained and saved the same way as in Model_Building.
"""
import numpy as np
import pandas as pd
import pytest

from src.models.predict import FraudPredictor


@pytest.fixture
def predictor(production_bundle):
    path, _ = production_bundle
    return FraudPredictor(path)


@pytest.fixture
def X_raw(production_bundle):
    _, X_test = production_bundle
    return X_test


def test_predict_proba_shape_and_range(predictor, X_raw):
    proba = predictor.predict_proba(X_raw)
    assert proba.shape == (len(X_raw),)
    assert np.all((proba >= 0) & (proba <= 1))


def test_predict_shape_and_values(predictor, X_raw):
    preds = predictor.predict(X_raw)
    assert preds.shape == (len(X_raw),)
    assert set(np.unique(preds)) <= {0, 1}


def test_single_row(predictor, X_raw):
    proba = predictor.predict_proba(X_raw.iloc[:1])
    assert proba.shape == (1,)


def test_dict_input_matches_dataframe(predictor, X_raw):
    records = X_raw.head(5).to_dict(orient='records')
    assert np.allclose(predictor.predict_proba(records), predictor.predict_proba(X_raw.head(5)))


def test_column_order_does_not_matter(predictor, X_raw):
    shuffled = X_raw[list(reversed(X_raw.columns))]
    assert np.allclose(predictor.predict_proba(shuffled), predictor.predict_proba(X_raw))


def test_batch_results(predictor, X_raw):
    results = predictor.predict_batch(X_raw.head(7).to_dict(orient='records'))
    assert len(results) == 7
    for r in results:
        assert 0 <= r['fraud_probability'] <= 1
        assert r['prediction'] in (0, 1)
        assert r['prediction_label'] == ('Fraud' if r['prediction'] else 'Legitimate')
        assert r['risk_level'] in ('Low', 'Medium', 'High', 'Critical')


def test_threshold_controls_predictions(predictor, X_raw):
    predictor.set_threshold(0.0)
    assert predictor.predict(X_raw).all()
    predictor.set_threshold(1.0)
    proba = predictor.predict_proba(X_raw)
    assert (predictor.predict(X_raw) == (proba >= 1.0)).all()


def test_learns_planted_signal(predictor, X_raw, prepared_fraud):
    """The synthetic fraud signal is strong; a working pipeline should rank fraud higher."""
    _, y = prepared_fraud
    proba = predictor.predict_proba(X_raw)
    y_test = y.loc[X_raw.index]
    assert proba[y_test.values == 1].mean() > proba[y_test.values == 0].mean() + 0.3


def test_unseen_categories(predictor, X_raw):
    row = X_raw.iloc[:1].copy()
    row['country'] = 'Atlantis'
    row['browser'] = 'Netscape'
    assert predictor.predict_proba(row).shape == (1,)


def test_missing_feature_rejected(predictor, X_raw):
    with pytest.raises(ValueError, match='Missing features'):
        predictor.predict_proba(X_raw.drop(columns=['age']))
