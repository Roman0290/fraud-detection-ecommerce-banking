"""
Model training utilities with class imbalance handling and MLflow tracking.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import TomekLinks, RandomUnderSampler
from typing import Dict, Tuple, Optional, Any
import mlflow
import mlflow.sklearn
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating fraud detection models with class balancing."""
    
    @staticmethod
    def get_model(model_name: str, use_class_weight: bool = True, **kwargs):
        """
        Get a model instance with optional class balancing.
        
        Args:
            model_name: Name of the model
            use_class_weight: Whether to use class_weight='balanced'
            **kwargs: Additional model parameters
            
        Returns:
            Model instance
        """
        models = {
            'logistic_regression': lambda: LogisticRegression(
                C=kwargs.get('C', 1.0),
                solver=kwargs.get('solver', 'liblinear'),
                max_iter=kwargs.get('max_iter', 1000),
                class_weight='balanced' if use_class_weight else None,
                random_state=kwargs.get('random_state', 42)
            ),
            
            'decision_tree': lambda: DecisionTreeClassifier(
                max_depth=kwargs.get('max_depth', 10),
                min_samples_split=kwargs.get('min_samples_split', 10),
                min_samples_leaf=kwargs.get('min_samples_leaf', 5),
                class_weight='balanced' if use_class_weight else None,
                random_state=kwargs.get('random_state', 42)
            ),
            
            'random_forest': lambda: RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                min_samples_split=kwargs.get('min_samples_split', 10),
                min_samples_leaf=kwargs.get('min_samples_leaf', 5),
                class_weight='balanced' if use_class_weight else None,
                random_state=kwargs.get('random_state', 42),
                n_jobs=kwargs.get('n_jobs', -1)
            ),
            
            'gradient_boosting': lambda: GradientBoostingClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 5),
                min_samples_split=kwargs.get('min_samples_split', 10),
                min_samples_leaf=kwargs.get('min_samples_leaf', 5),
                random_state=kwargs.get('random_state', 42)
            ),
            
            'mlp': lambda: MLPClassifier(
                hidden_layer_sizes=kwargs.get('hidden_layer_sizes', (100, 50)),
                max_iter=kwargs.get('max_iter', 500),
                learning_rate_init=kwargs.get('learning_rate_init', 0.001),
                early_stopping=kwargs.get('early_stopping', True),
                validation_fraction=kwargs.get('validation_fraction', 0.1),
                random_state=kwargs.get('random_state', 42)
            )
        }
        
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
        
        logger.info(f"Creating {model_name} model (class_weight_balanced={use_class_weight})")
        return models[model_name]()


class ResamplingStrategy:
    """Handle class imbalance through resampling techniques."""
    
    @staticmethod
    def apply_resampling(X_train: pd.DataFrame, y_train: pd.Series, 
                        strategy: str = 'none', 
                        random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply resampling strategy to handle class imbalance.
        
        Args:
            X_train: Training features
            y_train: Training labels
            strategy: Resampling strategy ('none', 'smote', 'smote_tomek', 'smote_enn', 
                     'adasyn', 'undersample', 'combined')
            random_state: Random seed
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        if strategy == 'none':
            logger.info("No resampling applied")
            return X_train, y_train
        
        logger.info(f"Applying {strategy} resampling")
        logger.info(f"Original class distribution: {pd.Series(y_train).value_counts().to_dict()}")
        
        if strategy == 'smote':
            sampler = SMOTE(random_state=random_state, k_neighbors=5)
        elif strategy == 'smote_tomek':
            sampler = SMOTETomek(random_state=random_state)
        elif strategy == 'smote_enn':
            sampler = SMOTEENN(random_state=random_state)
        elif strategy == 'adasyn':
            sampler = ADASYN(random_state=random_state)
        elif strategy == 'undersample':
            sampler = RandomUnderSampler(random_state=random_state, sampling_strategy='auto')
        elif strategy == 'combined':
            # Combination of undersampling majority + SMOTE on minority
            sampler = SMOTETomek(random_state=random_state)
        else:
            raise ValueError(f"Unknown resampling strategy: {strategy}")
        
        try:
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            logger.info(f"Resampled class distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
            
            # Convert back to DataFrame/Series with proper columns
            if isinstance(X_train, pd.DataFrame):
                X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
            if isinstance(y_train, pd.Series):
                y_resampled = pd.Series(y_resampled, name=y_train.name)
            
            return X_resampled, y_resampled
        
        except Exception as e:
            logger.error(f"Error during resampling: {e}")
            logger.warning("Falling back to original data")
            return X_train, y_train


class ModelTrainer:
    """Train fraud detection models with MLflow tracking."""
    
    def __init__(self, experiment_name: str = "fraud_detection", 
                 tracking_uri: str = None):
        """
        Initialize ModelTrainer.
        
        Args:
            experiment_name: MLflow experiment name
            tracking_uri: MLflow tracking URI (None for local ./mlruns)
        """
        self.experiment_name = experiment_name
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created new MLflow experiment: {experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})")
            
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            logger.warning(f"Could not set up MLflow experiment: {e}")
    
    def train_model(self, model_name: str, 
                   X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series,
                   resampling: str = 'none',
                   use_class_weight: bool = True,
                   model_params: Dict = None,
                   mlflow_logging: bool = True) -> Tuple[Any, Dict[str, Any]]:
        """
        Train a model with optional resampling and MLflow tracking.
        
        Args:
            model_name: Name of model to train
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            resampling: Resampling strategy
            use_class_weight: Whether to use class_weight='balanced'
            model_params: Additional model parameters
            mlflow_logging: Whether to log to MLflow
            
        Returns:
            Tuple of (trained_model, metrics_dict)
        """
        if model_params is None:
            model_params = {}
        
        # Start MLflow run
        if mlflow_logging:
            run = mlflow.start_run(run_name=f"{model_name}_{resampling}")
        
        try:
            # Apply resampling
            X_train_resampled, y_train_resampled = ResamplingStrategy.apply_resampling(
                X_train, y_train, strategy=resampling
            )
            
            # Create model
            model = ModelFactory.get_model(model_name, use_class_weight=use_class_weight, 
                                          **model_params)
            
            # Log parameters
            if mlflow_logging:
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("resampling", resampling)
                mlflow.log_param("use_class_weight", use_class_weight)
                mlflow.log_param("train_samples", len(X_train_resampled))
                mlflow.log_param("test_samples", len(X_test))
                mlflow.log_param("n_features", X_train.shape[1])
                
                for key, value in model_params.items():
                    mlflow.log_param(key, value)
            
            # Train model
            logger.info(f"Training {model_name}...")
            model.fit(X_train_resampled, y_train_resampled)
            logger.info("Training complete")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = None
            
            # Evaluate
            from scripts.models.evaluate import ModelEvaluator
            evaluator = ModelEvaluator(model_name)
            metrics = evaluator.evaluate_classification(y_test, y_pred, y_pred_proba)
            
            # Log metrics to MLflow
            if mlflow_logging:
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float, np.integer, np.floating)):
                        mlflow.log_metric(metric_name, float(metric_value))
                
                # Log model
                mlflow.sklearn.log_model(model, "model")
            
            return model, metrics
        
        finally:
            if mlflow_logging:
                mlflow.end_run()
    
    def train_multiple_models(self, model_configs: Dict[str, Dict],
                            X_train: pd.DataFrame, y_train: pd.Series,
                            X_test: pd.DataFrame, y_test: pd.Series,
                            output_dir: str = "models") -> Dict[str, Tuple[Any, Dict]]:
        """
        Train multiple models with different configurations.
        
        Args:
            model_configs: Dictionary mapping config names to model configurations
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            output_dir: Directory to save models
            
        Returns:
            Dictionary mapping config names to (model, metrics) tuples
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        for config_name, config in model_configs.items():
            logger.info(f"\n{'='*80}")
            logger.info(f"Training configuration: {config_name}")
            logger.info(f"{'='*80}")
            
            model, metrics = self.train_model(
                model_name=config.get('model_name'),
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                resampling=config.get('resampling', 'none'),
                use_class_weight=config.get('use_class_weight', True),
                model_params=config.get('params', {}),
                mlflow_logging=config.get('mlflow_logging', True)
            )
            
            # Save model
            model_path = os.path.join(output_dir, f"{config_name}.pkl")
            joblib.dump(model, model_path)
            logger.info(f"Model saved to {model_path}")
            
            results[config_name] = (model, metrics)
        
        # Compare results
        from scripts.models.evaluate import compare_models
        metrics_only = {name: metrics for name, (_, metrics) in results.items()}
        comparison_df = compare_models(metrics_only)
        
        # Save comparison
        comparison_path = os.path.join(output_dir, "model_comparison.csv")
        comparison_df.to_csv(comparison_path)
        logger.info(f"Model comparison saved to {comparison_path}")
        
        return results


def save_model(model: Any, model_path: str, metadata: Dict = None):
    """
    Save model with metadata.
    
    Args:
        model: Trained model
        model_path: Path to save model
        metadata: Optional metadata dictionary
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    model_data = {
        'model': model,
        'metadata': metadata or {}
    }
    
    joblib.dump(model_data, model_path)
    logger.info(f"Model saved to {model_path}")


def load_model(model_path: str) -> Tuple[Any, Dict]:
    """
    Load model with metadata.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Tuple of (model, metadata)
    """
    model_data = joblib.load(model_path)
    
    if isinstance(model_data, dict):
        return model_data['model'], model_data.get('metadata', {})
    else:
        # Legacy format (just the model)
        return model_data, {}


if __name__ == "__main__":
    # Test training
    from scripts.data.loader import DataLoader
    from scripts.features.engineering import FeatureEngineer, prepare_fraud_data
    from sklearn.model_selection import train_test_split
    
    # Load data
    loader = DataLoader()
    fraud_data = loader.load_fraud_data(validate=False)
    
    # Prepare features
    X, y = prepare_fraud_data(fraud_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                         random_state=42, stratify=y)
    
    # Engineer features
    engineer = FeatureEngineer()
    X_train, X_test = engineer.prepare_features(X_train, X_test)
    
    # Train a test model
    trainer = ModelTrainer(experiment_name="test_fraud_detection")
    model, metrics = trainer.train_model(
        model_name='logistic_regression',
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        resampling='smote',
        use_class_weight=True
    )
    
    print(f"\nTest training completed. F1 Score: {metrics['f1_score']:.4f}")
