"""
Model evaluation utilities with comprehensive metrics for fraud detection.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from typing import Dict, Tuple, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation for fraud detection."""
    
    def __init__(self, model_name: str = "Model"):
        """
        Initialize evaluator.
        
        Args:
            model_name: Name of the model being evaluated
        """
        self.model_name = model_name
        self.metrics = {}
        
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Comprehensive classification evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for positive class)
            
        Returns:
            Dictionary containing all metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        if len(cm) == 2:
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
            
            # Additional metrics
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Probability-based metrics
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
            
            # Store for plotting
            self.y_true = y_true
            self.y_pred_proba = y_pred_proba
        
        self.metrics = metrics
        
        # Log metrics
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluation Results for {self.model_name}")
        logger.info(f"{'='*60}")
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1 Score:  {metrics['f1_score']:.4f}")
        
        if 'roc_auc' in metrics:
            logger.info(f"ROC AUC:   {metrics['roc_auc']:.4f}")
            logger.info(f"PR AUC:    {metrics['average_precision']:.4f}")
        
        if 'true_positives' in metrics:
            logger.info(f"\nConfusion Matrix:")
            logger.info(f"  TN: {metrics['true_negatives']:6d}  |  FP: {metrics['false_positives']:6d}")
            logger.info(f"  FN: {metrics['false_negatives']:6d}  |  TP: {metrics['true_positives']:6d}")
        
        logger.info(f"{'='*60}\n")
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             save_path: Optional[str] = None, 
                             figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Plot confusion matrix heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Normal', 'Fraud'],
                   yticklabels=['Normal', 'Fraud'])
        
        ax.set_title(f'Confusion Matrix - {self.model_name}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                      save_path: Optional[str] = None,
                      figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {self.model_name}')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        return fig
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                    save_path: Optional[str] = None,
                                    figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Plot Precision-Recall curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(recall, precision, color='darkorange', lw=2,
               label=f'PR curve (AP = {avg_precision:.3f})')
        
        # Baseline (random classifier)
        baseline = np.sum(y_true) / len(y_true)
        ax.plot([0, 1], [baseline, baseline], color='navy', lw=2, 
               linestyle='--', label=f'Baseline ({baseline:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve - {self.model_name}')
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curve saved to {save_path}")
        
        return fig
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                              metric: str = 'f1', 
                              min_precision: Optional[float] = None) -> Dict[str, Any]:
        """
        Find optimal classification threshold.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            metric: Metric to optimize ('f1', 'precision', 'recall', 'youden')
            min_precision: Minimum precision constraint (for fraud detection)
            
        Returns:
            Dictionary with optimal threshold and metrics
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        # Calculate F1 scores for each threshold
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        
        if min_precision is not None:
            # Filter thresholds that meet minimum precision
            valid_indices = precisions >= min_precision
            if not np.any(valid_indices):
                logger.warning(f"No threshold achieves min_precision={min_precision}")
                valid_indices = np.ones_like(precisions, dtype=bool)
            
            f1_scores = np.where(valid_indices, f1_scores, -np.inf)
        
        if metric == 'f1':
            optimal_idx = np.argmax(f1_scores)
        elif metric == 'precision':
            optimal_idx = np.argmax(precisions)
        elif metric == 'recall':
            optimal_idx = np.argmax(recalls)
        elif metric == 'youden':
            # Youden's J statistic = sensitivity + specificity - 1
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = roc_thresholds[optimal_idx]
            
            # Recalculate metrics at this threshold
            y_pred_opt = (y_pred_proba >= optimal_threshold).astype(int)
            return {
                'threshold': optimal_threshold,
                'precision': precision_score(y_true, y_pred_opt, zero_division=0),
                'recall': recall_score(y_true, y_pred_opt, zero_division=0),
                'f1_score': f1_score(y_true, y_pred_opt, zero_division=0)
            }
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        optimal_threshold = thresholds[optimal_idx]
        
        result = {
            'threshold': optimal_threshold,
            'precision': precisions[optimal_idx],
            'recall': recalls[optimal_idx],
            'f1_score': f1_scores[optimal_idx]
        }
        
        logger.info(f"\nOptimal Threshold (optimizing {metric}):")
        logger.info(f"  Threshold: {result['threshold']:.4f}")
        logger.info(f"  Precision: {result['precision']:.4f}")
        logger.info(f"  Recall:    {result['recall']:.4f}")
        logger.info(f"  F1 Score:  {result['f1_score']:.4f}")
        
        return result
    
    def evaluate_at_precision_target(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                    target_precision: float = 0.85) -> Dict[str, Any]:
        """
        Evaluate model at a target precision level (business requirement).
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            target_precision: Target precision threshold
            
        Returns:
            Dictionary with threshold and metrics
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        # Find threshold that achieves target precision
        valid_indices = precisions >= target_precision
        
        if not np.any(valid_indices):
            logger.warning(f"Cannot achieve target precision {target_precision}")
            # Use highest precision achievable
            optimal_idx = np.argmax(precisions)
        else:
            # Among valid thresholds, choose one with highest recall
            valid_recalls = np.where(valid_indices, recalls, -np.inf)
            optimal_idx = np.argmax(valid_recalls)
        
        optimal_threshold = thresholds[optimal_idx]
        y_pred_opt = (y_pred_proba >= optimal_threshold).astype(int)
        
        result = {
            'threshold': optimal_threshold,
            'precision': precision_score(y_true, y_pred_opt, zero_division=0),
            'recall': recall_score(y_true, y_pred_opt, zero_division=0),
            'f1_score': f1_score(y_true, y_pred_opt, zero_division=0),
            'fraud_caught': int(np.sum((y_true == 1) & (y_pred_opt == 1))),
            'fraud_missed': int(np.sum((y_true == 1) & (y_pred_opt == 0))),
            'false_alarms': int(np.sum((y_true == 0) & (y_pred_opt == 1)))
        }
        
        logger.info(f"\nEvaluation at Precision Target ({target_precision}):")
        logger.info(f"  Threshold:    {result['threshold']:.4f}")
        logger.info(f"  Precision:    {result['precision']:.4f}")
        logger.info(f"  Recall:       {result['recall']:.4f}")
        logger.info(f"  F1 Score:     {result['f1_score']:.4f}")
        logger.info(f"  Fraud Caught: {result['fraud_caught']}")
        logger.info(f"  Fraud Missed: {result['fraud_missed']}")
        logger.info(f"  False Alarms: {result['false_alarms']}")
        
        return result
    
    def generate_full_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                           y_pred_proba: Optional[np.ndarray] = None,
                           output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report with all metrics and plots.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            output_dir: Directory to save plots
            
        Returns:
            Dictionary with all evaluation results
        """
        # Basic metrics
        metrics = self.evaluate_classification(y_true, y_pred, y_pred_proba)
        
        # Classification report
        metrics['classification_report'] = classification_report(y_true, y_pred, 
                                                                 target_names=['Normal', 'Fraud'],
                                                                 output_dict=True)
        
        # Generate plots if output directory provided
        if output_dir and y_pred_proba is not None:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            # Confusion matrix
            self.plot_confusion_matrix(y_true, y_pred, 
                                      save_path=os.path.join(output_dir, f'{self.model_name}_confusion_matrix.png'))
            
            # ROC curve
            self.plot_roc_curve(y_true, y_pred_proba,
                               save_path=os.path.join(output_dir, f'{self.model_name}_roc_curve.png'))
            
            # PR curve
            self.plot_precision_recall_curve(y_true, y_pred_proba,
                                            save_path=os.path.join(output_dir, f'{self.model_name}_pr_curve.png'))
            
            # Threshold analysis
            metrics['optimal_threshold_f1'] = self.find_optimal_threshold(y_true, y_pred_proba, metric='f1')
            metrics['precision_target_85'] = self.evaluate_at_precision_target(y_true, y_pred_proba, 
                                                                               target_precision=0.85)
        
        return metrics


def compare_models(results: Dict[str, Dict[str, Any]], 
                  metrics: list = None) -> pd.DataFrame:
    """
    Compare multiple model results.
    
    Args:
        results: Dictionary mapping model names to their metric dictionaries
        metrics: List of metrics to compare (None for all)
        
    Returns:
        DataFrame with comparison
    """
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'average_precision']
    
    comparison = {}
    for model_name, model_metrics in results.items():
        comparison[model_name] = {k: model_metrics.get(k, np.nan) for k in metrics}
    
    df = pd.DataFrame(comparison).T
    df = df.round(4)
    
    logger.info("\nModel Comparison:")
    logger.info("\n" + df.to_string())
    
    return df


if __name__ == "__main__":
    # Test with dummy data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_pred_proba = np.random.rand(1000)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    evaluator = ModelEvaluator("Test Model")
    metrics = evaluator.evaluate_classification(y_true, y_pred, y_pred_proba)
    print(f"\nTest completed. Metrics: {metrics}")
