"""
Model explainability utilities using SHAP and LIME.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from lime import lime_tabular
from typing import Any, Optional, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelExplainer:
    """Provide model explanations using SHAP and LIME."""
    
    def __init__(self, model: Any, X_train: pd.DataFrame, 
                 feature_names: List[str] = None):
        """
        Initialize explainer.
        
        Args:
            model: Trained model
            X_train: Training data (for background samples)
            feature_names: List of feature names
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or (X_train.columns.tolist() if hasattr(X_train, 'columns') else None)
        self.shap_explainer = None
        self.lime_explainer = None
        
    def initialize_shap(self, background_samples: int = 100):
        """
        Initialize SHAP explainer.
        
        Args:
            background_samples: Number of background samples for SHAP
        """
        try:
            logger.info("Initializing SHAP explainer...")
            
            # Select appropriate explainer based on model type
            model_type = type(self.model).__name__
            
            if 'Tree' in model_type or 'Forest' in model_type or 'Gradient' in model_type:
                # Tree-based models
                self.shap_explainer = shap.TreeExplainer(self.model)
                logger.info(f"Using TreeExplainer for {model_type}")
            else:
                # Other models (use KernelExplainer with background samples)
                background = shap.sample(self.X_train, background_samples)
                self.shap_explainer = shap.KernelExplainer(
                    self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                    background
                )
                logger.info(f"Using KernelExplainer for {model_type}")
            
            logger.info("SHAP explainer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing SHAP: {e}")
            self.shap_explainer = None
    
    def initialize_lime(self, mode: str = 'classification'):
        """
        Initialize LIME explainer.
        
        Args:
            mode: 'classification' or 'regression'
        """
        try:
            logger.info("Initializing LIME explainer...")
            
            self.lime_explainer = lime_tabular.LimeTabularExplainer(
                training_data=self.X_train.values if hasattr(self.X_train, 'values') else self.X_train,
                feature_names=self.feature_names,
                class_names=['Normal', 'Fraud'],
                mode=mode,
                random_state=42
            )
            
            logger.info("LIME explainer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LIME: {e}")
            self.lime_explainer = None
    
    def explain_with_shap(self, X_explain: pd.DataFrame, 
                         plot_type: str = 'summary',
                         save_path: Optional[str] = None,
                         max_display: int = 20) -> Any:
        """
        Generate SHAP explanations.
        
        Args:
            X_explain: Data to explain
            plot_type: Type of plot ('summary', 'waterfall', 'force', 'dependence')
            save_path: Path to save the plot
            max_display: Maximum number of features to display
            
        Returns:
            SHAP values
        """
        if self.shap_explainer is None:
            self.initialize_shap()
        
        if self.shap_explainer is None:
            logger.error("SHAP explainer not available")
            return None
        
        try:
            logger.info(f"Calculating SHAP values for {len(X_explain)} instances...")
            shap_values = self.shap_explainer.shap_values(X_explain)
            
            # Handle multi-output (classification) case
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values_fraud = shap_values[1]  # Fraud class
            else:
                shap_values_fraud = shap_values
            
            # Generate plot
            if plot_type == 'summary':
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values_fraud, X_explain, 
                                 feature_names=self.feature_names,
                                 max_display=max_display, show=False)
                plt.tight_layout()
                
            elif plot_type == 'waterfall' and len(X_explain) > 0:
                plt.figure(figsize=(10, 8))
                if isinstance(shap_values, list):
                    shap.waterfall_plot(shap.Explanation(
                        values=shap_values[1][0],
                        base_values=self.shap_explainer.expected_value[1] if isinstance(self.shap_explainer.expected_value, list) else self.shap_explainer.expected_value,
                        data=X_explain.iloc[0].values,
                        feature_names=self.feature_names
                    ), max_display=max_display, show=False)
                else:
                    shap.waterfall_plot(shap.Explanation(
                        values=shap_values[0],
                        base_values=self.shap_explainer.expected_value,
                        data=X_explain.iloc[0].values,
                        feature_names=self.feature_names
                    ), max_display=max_display, show=False)
                plt.tight_layout()
                
            elif plot_type == 'bar':
                plt.figure(figsize=(10, 8))
                shap.plots.bar(shap_values_fraud, max_display=max_display, show=False)
                plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"SHAP plot saved to {save_path}")
                plt.close()
            else:
                plt.show()
            
            return shap_values
        
        except Exception as e:
            logger.error(f"Error generating SHAP explanations: {e}")
            return None
    
    def explain_with_lime(self, instance: np.ndarray, 
                         instance_index: int = 0,
                         num_features: int = 10,
                         save_path: Optional[str] = None) -> Any:
        """
        Generate LIME explanation for a single instance.
        
        Args:
            instance: Single instance to explain
            instance_index: Index for labeling
            num_features: Number of top features to show
            save_path: Path to save the explanation
            
        Returns:
            LIME explanation object
        """
        if self.lime_explainer is None:
            self.initialize_lime()
        
        if self.lime_explainer is None:
            logger.error("LIME explainer not available")
            return None
        
        try:
            logger.info(f"Generating LIME explanation for instance {instance_index}...")
            
            # Get prediction function
            if hasattr(self.model, 'predict_proba'):
                predict_fn = self.model.predict_proba
            else:
                predict_fn = lambda x: np.column_stack([1 - self.model.predict(x), self.model.predict(x)])
            
            # Generate explanation
            explanation = self.lime_explainer.explain_instance(
                data_row=instance.flatten() if isinstance(instance, np.ndarray) else instance,
                predict_fn=predict_fn,
                num_features=num_features
            )
            
            # Visualize
            fig = explanation.as_pyplot_figure()
            plt.title(f'LIME Explanation - Instance {instance_index}')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"LIME explanation saved to {save_path}")
                plt.close()
            else:
                plt.show()
            
            return explanation
        
        except Exception as e:
            logger.error(f"Error generating LIME explanation: {e}")
            return None
    
    def get_feature_importance(self, method: str = 'shap', 
                              X_sample: pd.DataFrame = None,
                              top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance using SHAP or model's built-in importance.
        
        Args:
            method: 'shap' or 'model'
            X_sample: Sample data for SHAP (if None, uses X_train sample)
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if method == 'shap':
            if X_sample is None:
                X_sample = self.X_train.sample(min(100, len(self.X_train)))
            
            if self.shap_explainer is None:
                self.initialize_shap()
            
            if self.shap_explainer is None:
                logger.error("SHAP explainer not available")
                return None
            
            shap_values = self.shap_explainer.shap_values(X_sample)

            # Handle multi-output case
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Fraud class

            # Calculate mean absolute SHAP values
            importance = np.abs(shap_values).mean(axis=0)

            # Ensure importance is 1D
            if importance.ndim > 1:
                importance = importance.flatten()

            # Validate shapes
            if len(importance) != len(self.feature_names):
                logger.warning(f"Feature name count ({len(self.feature_names)}) != importance count ({len(importance)})")
                # Trim to match
                min_len = min(len(importance), len(self.feature_names))
                importance = importance[:min_len]
                feature_names = self.feature_names[:min_len]
            else:
                feature_names = self.feature_names
            
        elif method == 'model':
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                importance = np.abs(self.model.coef_[0])
            else:
                logger.error("Model does not have feature importance")
                return None

            # Ensure feature_names is set for model method
            feature_names = self.feature_names
        else:
            raise ValueError(f"Unknown method: {method}")

        # Ensure importance is 1D for model method too
        if importance.ndim > 1:
            importance = importance.flatten()

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        logger.info(f"\nTop {top_n} features ({method}):")
        logger.info("\n" + importance_df.to_string(index=False))
        
        return importance_df
    
    def plot_feature_importance(self, method: str = 'shap',
                               X_sample: pd.DataFrame = None,
                               top_n: int = 20,
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (10, 8)):
        """
        Plot feature importance.
        
        Args:
            method: 'shap' or 'model'
            X_sample: Sample data for SHAP
            top_n: Number of top features
            save_path: Path to save plot
            figsize: Figure size
        """
        importance_df = self.get_feature_importance(method, X_sample, top_n)
        
        if importance_df is None:
            return
        
        plt.figure(figsize=figsize)
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Importance')
        plt.title(f'Feature Importance ({method.upper()})')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
            plt.close()
        else:
            plt.show()


def explain_model_predictions(model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame,
                              output_dir: str = "explanations",
                              num_instances: int = 5):
    """
    Generate comprehensive explanations for a model.
    
    Args:
        model: Trained model
        X_train: Training data
        X_test: Test data
        output_dir: Directory to save explanations
        num_instances: Number of individual instances to explain with LIME
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    explainer = ModelExplainer(model, X_train)
    
    # SHAP summary plot
    logger.info("Generating SHAP summary plot...")
    explainer.explain_with_shap(
        X_test.sample(min(100, len(X_test))),
        plot_type='summary',
        save_path=os.path.join(output_dir, 'shap_summary.png')
    )
    
    # Feature importance
    logger.info("Generating feature importance plots...")
    explainer.plot_feature_importance(
        method='shap',
        save_path=os.path.join(output_dir, 'feature_importance_shap.png')
    )
    
    if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
        explainer.plot_feature_importance(
            method='model',
            save_path=os.path.join(output_dir, 'feature_importance_model.png')
        )
    
    # LIME explanations for individual instances
    logger.info(f"Generating LIME explanations for {num_instances} instances...")
    for i in range(min(num_instances, len(X_test))):
        instance = X_test.iloc[i:i+1].values[0]
        explainer.explain_with_lime(
            instance,
            instance_index=i,
            save_path=os.path.join(output_dir, f'lime_instance_{i}.png')
        )
    
    logger.info(f"Explanations saved to {output_dir}")


if __name__ == "__main__":
    # Test explainability
    from scripts.data.loader import DataLoader
    from scripts.features.engineering import FeatureEngineer, prepare_fraud_data
    from scripts.models.train import ModelFactory
    from sklearn.model_selection import train_test_split
    
    # Load and prepare data
    loader = DataLoader()
    fraud_data = loader.load_fraud_data(validate=False)
    X, y = prepare_fraud_data(fraud_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    engineer = FeatureEngineer()
    X_train, X_test = engineer.prepare_features(X_train, X_test)
    
    # Train a simple model
    model = ModelFactory.get_model('random_forest', use_class_weight=True)
    model.fit(X_train, y_train)
    
    # Explain
    explainer = ModelExplainer(model, X_train)
    importance_df = explainer.get_feature_importance(method='model')
    print(f"\nTest completed. Top feature: {importance_df.iloc[0]['feature']}")
