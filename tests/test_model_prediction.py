"""
Tests for model prediction functionality.
"""
import pytest
import pandas as pd
import numpy as np
import joblib
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.predict import FraudPredictor, ModelRegistry
from sklearn.ensemble import RandomForestClassifier


class TestFraudPredictor:
    """Test FraudPredictor class."""
    
    @pytest.fixture
    def trained_model(self, tmp_path):
        """Create and save a simple trained model."""
        # Create dummy data
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 2, 100)
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        joblib.dump(model, model_path)
        
        return str(model_path)
    
    def test_load_model(self, trained_model):
        """Test model loading."""
        predictor = FraudPredictor(trained_model)
        
        assert predictor.model is not None
        assert isinstance(predictor.model, RandomForestClassifier)
    
    def test_predict_proba(self, trained_model):
        """Test probability prediction."""
        predictor = FraudPredictor(trained_model)
        
        # Create test data
        X_test = np.random.rand(10, 5)
        probabilities = predictor.predict_proba(X_test)
        
        assert len(probabilities) == 10
        assert all(0 <= p <= 1 for p in probabilities)
    
    def test_predict(self, trained_model):
        """Test binary prediction."""
        predictor = FraudPredictor(trained_model, threshold=0.5)
        
        # Create test data
        X_test = np.random.rand(10, 5)
        predictions = predictor.predict(X_test)
        
        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)
    
    def test_predict_single(self, trained_model):
        """Test single transaction prediction."""
        predictor = FraudPredictor(trained_model)
        
        # Create test transaction
        transaction = {f'feature_{i}': np.random.rand() for i in range(5)}
        
        result = predictor.predict_single(transaction)
        
        assert 'prediction' in result
        assert 'fraud_probability' in result
        assert 'prediction_label' in result
        assert 'risk_level' in result
        assert result['prediction'] in [0, 1]
    
    def test_predict_batch(self, trained_model):
        """Test batch prediction."""
        predictor = FraudPredictor(trained_model)
        
        # Create test transactions
        transactions = [
            {f'feature_{i}': np.random.rand() for i in range(5)}
            for _ in range(5)
        ]
        
        results = predictor.predict_batch(transactions)
        
        assert len(results) == 5
        assert all('prediction' in r for r in results)
        assert all('fraud_probability' in r for r in results)
    
    def test_set_threshold(self, trained_model):
        """Test threshold updating."""
        predictor = FraudPredictor(trained_model, threshold=0.5)
        
        predictor.set_threshold(0.7)
        assert predictor.threshold == 0.7
        
        with pytest.raises(ValueError):
            predictor.set_threshold(1.5)
    
    def test_risk_level(self, trained_model):
        """Test risk level categorization."""
        predictor = FraudPredictor(trained_model)
        
        assert predictor._get_risk_level(0.2) == 'Low'
        assert predictor._get_risk_level(0.5) == 'Medium'
        assert predictor._get_risk_level(0.7) == 'High'
        assert predictor._get_risk_level(0.9) == 'Critical'


class TestModelRegistry:
    """Test ModelRegistry class."""
    
    @pytest.fixture
    def models_dir(self, tmp_path):
        """Create temporary models directory with test models."""
        # Create and save two test models
        for i in range(2):
            X_train = np.random.rand(50, 5)
            y_train = np.random.randint(0, 2, 50)
            
            model = RandomForestClassifier(n_estimators=5, random_state=i)
            model.fit(X_train, y_train)
            
            model_path = tmp_path / f"model_{i}.pkl"
            joblib.dump(model, model_path)
        
        return str(tmp_path)
    
    def test_load_model(self, models_dir):
        """Test loading model into registry."""
        registry = ModelRegistry(models_dir=models_dir)
        
        predictor = registry.load_model('model_0', 'model_0.pkl')
        
        assert isinstance(predictor, FraudPredictor)
        assert 'model_0' in registry.list_models()
    
    def test_get_model(self, models_dir):
        """Test retrieving model from registry."""
        registry = ModelRegistry(models_dir=models_dir)
        registry.load_model('model_0', 'model_0.pkl')
        
        predictor = registry.get_model('model_0')
        
        assert isinstance(predictor, FraudPredictor)
    
    def test_default_model(self, models_dir):
        """Test default model setting."""
        registry = ModelRegistry(models_dir=models_dir)
        registry.load_model('model_0', 'model_0.pkl')
        registry.load_model('model_1', 'model_1.pkl')
        
        # First loaded should be default
        assert registry.default_model == 'model_0'
        
        # Change default
        registry.set_default_model('model_1')
        assert registry.default_model == 'model_1'
    
    def test_list_models(self, models_dir):
        """Test listing loaded models."""
        registry = ModelRegistry(models_dir=models_dir)
        registry.load_model('model_0', 'model_0.pkl')
        registry.load_model('model_1', 'model_1.pkl')
        
        models = registry.list_models()
        
        assert len(models) == 2
        assert 'model_0' in models
        assert 'model_1' in models


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
