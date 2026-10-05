"""
Model loading and inference for fraud detection.
"""
import os
import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Upper bounds (exclusive) of the Low / Medium / High risk bands; anything above is Critical
RISK_LEVELS = [(0.3, 'Low'), (0.6, 'Medium'), (0.8, 'High')]


def _load_artifact(path: str) -> Dict[str, Any]:
    """
    Load a saved model file.

    Accepts a bare estimator (as saved by ModelTrainer.train_multiple_models) or a
    bundle dict with 'model' and optional 'transformer' and 'metadata' keys (as saved
    by ModelRegistry.save_model or scripts.models.train.save_model).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    artifact = joblib.load(path)
    if isinstance(artifact, dict) and 'model' in artifact:
        return {
            'model': artifact['model'],
            'transformer': artifact.get('transformer'),
            'metadata': artifact.get('metadata', {}),
        }
    return {'model': artifact, 'transformer': None, 'metadata': {}}


class FraudPredictor:
    """Load a trained model and score transactions."""

    def __init__(self, model_path: str, threshold: float = 0.5,
                 transformer_path: Optional[str] = None):
        """
        Initialize FraudPredictor.

        Args:
            model_path: Path to a saved model or model bundle
            threshold: Fraud probability at or above which a transaction is flagged
            transformer_path: Optional path to a fitted FeatureEngineer. Overrides a
                transformer stored in the model bundle.
        """
        artifact = _load_artifact(model_path)
        self.model_path = model_path
        self.model = artifact['model']
        self.transformer = artifact['transformer']
        self.metadata = artifact['metadata']

        if transformer_path:
            self.transformer = joblib.load(transformer_path)

        if not hasattr(self.model, 'predict_proba'):
            raise TypeError(f"Model {type(self.model).__name__} does not support predict_proba")

        self.threshold = None
        self.set_threshold(threshold)

        logger.info(f"Loaded {type(self.model).__name__} from {model_path}"
                    f"{' with feature transformer' if self.transformer is not None else ''}")

    @property
    def feature_names(self) -> Optional[List[str]]:
        """Input columns the predictor expects, if known."""
        if self.transformer is not None:
            return self.metadata.get('input_features')
        names = getattr(self.model, 'feature_names_in_', None)
        return list(names) if names is not None else None

    def set_threshold(self, threshold: float):
        """Set the decision threshold (must be between 0 and 1)."""
        if not 0 <= threshold <= 1:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
        self.threshold = threshold

    def preprocess(self, X: Union[pd.DataFrame, np.ndarray, Dict, List[Dict]]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Convert input into the feature matrix the model was trained on.

        Args:
            X: DataFrame, array, a single transaction dict or a list of dicts

        Returns:
            Model-ready features
        """
        if isinstance(X, dict):
            X = [X]
        if isinstance(X, list):
            X = pd.DataFrame(X)

        if self.transformer is not None:
            X = pd.DataFrame(X)
            expected = self.feature_names or []
            missing = [c for c in expected if c not in X.columns]
            if missing:
                raise ValueError(f"Missing features: {missing}")
            X = self.transformer.transform(X)

        if isinstance(X, pd.DataFrame):
            names = getattr(self.model, 'feature_names_in_', None)
            if names is None:
                return X.values
            missing = [c for c in names if c not in X.columns]
            if missing:
                raise ValueError(f"Missing features: {missing}")
            return X[list(names)]

        return X

    def predict_proba(self, X) -> np.ndarray:
        """Return the fraud probability for each row."""
        return self.model.predict_proba(self.preprocess(X))[:, 1]

    def predict(self, X) -> np.ndarray:
        """Return 1 (fraud) or 0 (legitimate) for each row using the current threshold."""
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def _get_risk_level(self, probability: float) -> str:
        """Map a fraud probability to a risk band."""
        for upper, level in RISK_LEVELS:
            if probability < upper:
                return level
        return 'Critical'

    def _format_result(self, probability: float) -> Dict[str, Any]:
        prediction = int(probability >= self.threshold)
        return {
            'prediction': prediction,
            'prediction_label': 'Fraud' if prediction else 'Legitimate',
            'fraud_probability': float(probability),
            'risk_level': self._get_risk_level(probability),
            'threshold': self.threshold,
        }

    def predict_single(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Score one transaction."""
        return self._format_result(self.predict_proba(transaction)[0])

    def predict_batch(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score a list of transactions."""
        if not transactions:
            return []
        return [self._format_result(p) for p in self.predict_proba(transactions)]


class ModelRegistry:
    """Save, load and keep track of models in a directory."""

    def __init__(self, models_dir: str = 'models'):
        """
        Initialize ModelRegistry.

        Args:
            models_dir: Directory where model files live
        """
        self.models_dir = models_dir
        self.models: Dict[str, FraudPredictor] = {}
        self.default_model: Optional[str] = None

    def _path(self, filename: str) -> str:
        return filename if os.path.isabs(filename) else os.path.join(self.models_dir, filename)

    def save_model(self, model: Any, filename: str, transformer: Any = None,
                   metadata: Optional[Dict] = None) -> str:
        """
        Save a model (and optional fitted FeatureEngineer) as a bundle.

        Args:
            model: Trained estimator
            filename: File name inside models_dir (or an absolute path)
            transformer: Optional fitted FeatureEngineer
            metadata: Optional metadata, e.g. metrics or 'input_features'

        Returns:
            Path the bundle was written to
        """
        path = self._path(filename)
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        joblib.dump({'model': model, 'transformer': transformer, 'metadata': metadata or {}}, path)
        logger.info(f"Model saved to {path}")
        return path

    def load_model(self, name: str, filename: str, threshold: float = 0.5) -> FraudPredictor:
        """
        Load a model file and register it under a name.

        The first model loaded becomes the default.
        """
        predictor = FraudPredictor(self._path(filename), threshold=threshold)
        self.models[name] = predictor
        if self.default_model is None:
            self.default_model = name
        return predictor

    def get_model(self, name: Optional[str] = None) -> FraudPredictor:
        """Return a loaded model by name, or the default model."""
        name = name or self.default_model
        if name not in self.models:
            raise KeyError(f"Model '{name}' is not loaded")
        return self.models[name]

    def set_default_model(self, name: str):
        """Make a loaded model the default."""
        if name not in self.models:
            raise KeyError(f"Model '{name}' is not loaded")
        self.default_model = name

    def list_models(self) -> List[str]:
        """Names of loaded models."""
        return list(self.models)

    def available_models(self) -> List[str]:
        """Model files (.pkl / .joblib) in models_dir."""
        if not os.path.isdir(self.models_dir):
            return []
        return sorted(f for f in os.listdir(self.models_dir) if f.endswith(('.pkl', '.joblib')))
