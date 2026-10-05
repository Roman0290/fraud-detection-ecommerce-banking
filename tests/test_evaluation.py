"""
Tests for model evaluation and explainability helpers.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from scripts.models.evaluate import ModelEvaluator, compare_models
from scripts.models.explainability import ModelExplainer


@pytest.fixture
def one_feature_data():
    """Only column 'c' determines the label."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((300, 4)), columns=['a', 'b', 'c', 'd'])
    y = (X['c'] > 0.7).astype(int)
    return X, y


class TestThresholds:
    # Scores that never reach 95% precision at any real threshold
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    proba = np.array([.9, .8, .7, .6, .5, .4, .3, .2])

    def test_unreachable_precision_target(self):
        result = ModelEvaluator().evaluate_at_precision_target(self.y, self.proba, target_precision=0.95)
        assert result['threshold'] in self.proba

    @pytest.mark.parametrize('metric', ['f1', 'precision', 'recall', 'youden'])
    def test_optimal_threshold_is_a_real_threshold(self, metric):
        result = ModelEvaluator().find_optimal_threshold(self.y, self.proba, metric=metric)
        assert self.proba.min() <= result['threshold'] <= self.proba.max()

    def test_f1_threshold_matches_reported_metrics(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        proba = np.array([.1, .2, .6, .4, .8, .9])
        result = ModelEvaluator().find_optimal_threshold(y, proba, metric='f1')
        pred = (proba >= result['threshold']).astype(int)
        assert result['recall'] == pytest.approx(pred[y == 1].mean())


def test_evaluate_classification_metrics():
    y = np.array([0, 0, 1, 1])
    metrics = ModelEvaluator().evaluate_classification(y, np.array([0, 1, 1, 1]), np.array([.1, .6, .7, .9]))
    assert metrics['precision'] == pytest.approx(2 / 3)
    assert metrics['recall'] == 1.0
    assert metrics['false_positives'] == 1
    assert metrics['roc_auc'] == 1.0


def test_compare_models():
    df = compare_models({'m1': {'f1_score': 0.5}, 'm2': {'f1_score': 0.7}}, metrics=['f1_score'])
    assert df['f1_score'].idxmax() == 'm2'


@pytest.mark.parametrize('model_cls', [RandomForestClassifier, GradientBoostingClassifier])
def test_shap_importance_finds_the_signal_feature(one_feature_data, model_cls):
    X, y = one_feature_data
    model = model_cls(n_estimators=20, random_state=0).fit(X, y)
    importance = ModelExplainer(model, X).get_feature_importance('shap', X_sample=X.head(100))
    assert len(importance) == 4
    assert importance.iloc[0]['feature'] == 'c'
    assert importance.iloc[0]['importance'] > 3 * importance.iloc[1]['importance']


def test_shap_returns_values_for_fraud_plots(one_feature_data):
    X, y = one_feature_data
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    explainer = ModelExplainer(model, X)
    for plot_type in ('summary', 'waterfall'):
        assert explainer.explain_with_shap(X.head(20), plot_type=plot_type) is not None
