"""
Flask API serving fraud predictions.

Configuration (environment variables):
    DEFAULT_MODEL_PATH        Model bundle to load (default: models/production_model.pkl)
    FEATURE_TRANSFORMER_PATH  Optional fitted FeatureEngineer, if not stored in the bundle
    MODEL_THRESHOLD           Decision threshold (default: 0.5)
"""
import os
import logging
from typing import Optional

from flask import Flask, jsonify, request

from src.models.predict import FraudPredictor

logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)


def load_predictor() -> Optional[FraudPredictor]:
    """Load the model from the environment config, or return None if it is unavailable."""
    model_path = os.environ.get('DEFAULT_MODEL_PATH', 'models/production_model.pkl')
    transformer_path = os.environ.get('FEATURE_TRANSFORMER_PATH')
    if transformer_path and not os.path.exists(transformer_path):
        transformer_path = None
    threshold = float(os.environ.get('MODEL_THRESHOLD', 0.5))

    try:
        return FraudPredictor(model_path, threshold=threshold, transformer_path=transformer_path)
    except FileNotFoundError:
        logger.warning(f"No model at {model_path}; /predict will return 503 until one is provided")
    except Exception:
        logger.exception(f"Failed to load model from {model_path}")
    return None


def create_app(predictor: Optional[FraudPredictor] = None) -> Flask:
    """
    Build the Flask app.

    Args:
        predictor: Predictor to serve. If None, one is loaded from the environment config.
    """
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))
    app.config['predictor'] = predictor if predictor is not None else load_predictor()

    @app.get('/health')
    def health():
        p = app.config['predictor']
        return jsonify({
            'status': 'ok',
            'model_loaded': p is not None,
            'model_path': p.model_path if p is not None else None,
        })

    @app.post('/predict')
    def predict():
        """
        Score transactions.

        Body: one transaction object, or {"transactions": [ ... ]}.
        Each transaction holds the model's input features.
        """
        p = app.config['predictor']
        if p is None:
            return jsonify({'error': 'Model not loaded'}), 503

        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({'error': 'Request body must be JSON'}), 400

        if isinstance(payload, dict) and 'transactions' in payload:
            transactions = payload['transactions']
            if not isinstance(transactions, list) or not all(isinstance(t, dict) for t in transactions):
                return jsonify({'error': "'transactions' must be a list of objects"}), 400
            single = False
        elif isinstance(payload, dict):
            transactions = [payload]
            single = True
        else:
            return jsonify({'error': 'Body must be a transaction object or {"transactions": [...]}'}), 400

        if not transactions:
            return jsonify({'predictions': []})

        try:
            results = p.predict_batch(transactions)
        except (ValueError, KeyError, TypeError) as e:
            return jsonify({'error': f'Invalid input: {e}'}), 400

        return jsonify(results[0] if single else {'predictions': results})

    return app


app = create_app()


if __name__ == '__main__':
    app.run(
        host=os.environ.get('API_HOST', '0.0.0.0'),
        port=int(os.environ.get('API_PORT', 5000)),
        debug=os.environ.get('API_DEBUG', 'false').lower() == 'true',
    )
