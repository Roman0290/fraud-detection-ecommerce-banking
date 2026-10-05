"""
Tests for the Flask API.
"""
import pytest

from src.api.app import create_app
from src.models.predict import FraudPredictor


@pytest.fixture
def client(production_bundle):
    path, _ = production_bundle
    return create_app(FraudPredictor(path)).test_client()


@pytest.fixture
def transactions(production_bundle):
    _, X_test = production_bundle
    return X_test.head(3).to_dict(orient='records')


def test_health(client):
    r = client.get('/health')
    assert r.status_code == 200
    assert r.get_json()['model_loaded'] is True


def test_predict_single(client, transactions):
    r = client.post('/predict', json=transactions[0])
    assert r.status_code == 200
    body = r.get_json()
    assert body['prediction'] in (0, 1)
    assert 0 <= body['fraud_probability'] <= 1


def test_predict_batch(client, transactions):
    r = client.post('/predict', json={'transactions': transactions})
    assert r.status_code == 200
    assert len(r.get_json()['predictions']) == 3


@pytest.mark.parametrize('kwargs', [
    {'data': 'not json', 'content_type': 'application/json'},
    {'json': [1, 2]},
    {'json': {'transactions': 'nope'}},
    {'json': {'purchase_value': 10}},
])
def test_bad_input_returns_400(client, kwargs):
    assert client.post('/predict', **kwargs).status_code == 400


def test_no_model_returns_503(monkeypatch, tmp_path):
    monkeypatch.setenv('DEFAULT_MODEL_PATH', str(tmp_path / 'missing.pkl'))
    client = create_app().test_client()
    assert client.get('/health').get_json()['model_loaded'] is False
    assert client.post('/predict', json={}).status_code == 503
