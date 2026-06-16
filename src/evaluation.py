"""Model evaluation metrics and results persistence."""

from datetime import datetime
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

RANDOM_STATE = 42


def _f2_score(precision: float, recall: float) -> float:
    """Compute F2 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return (5 * precision * recall) / (4 * precision + recall)


def apply_threshold(
    y_prob_alive: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Convert probability of Alive class to binary prediction.
    Predict Dead (0) if P(Dead) = 1 - y_prob_alive > threshold.
    Return array of 0/1 predictions.
    """
    y_prob_alive = np.asarray(y_prob_alive)
    p_dead = 1 - y_prob_alive
    return np.where(p_dead > threshold, 0, 1)


def get_all_metrics(
    model_name: str,
    y_true,
    y_pred,
    y_prob,
    threshold: float = 0.5,
) -> Dict[str, Union[str, float]]:
    """
    Compute and return a dict with keys:
      model_name, threshold, accuracy, recall_dead, precision_dead,
      f1_dead, f2_dead, roc_auc, pr_auc, false_negatives,
      recall_alive, precision_alive, f1_alive

    Dead class = 0, Alive class = 1.
    y_prob must be the probability of class 1 (Alive). Internally convert:
      P(Dead) = 1 - y_prob
    and apply threshold on P(Dead) > threshold to get y_pred_thresh.

    F2 score formula: (5 * precision * recall) / (4 * precision + recall)
    All float values rounded to 4 decimal places.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred_thresh = apply_threshold(y_prob, threshold=threshold)

    recall_dead = recall_score(y_true, y_pred_thresh, pos_label=0, zero_division=0)
    precision_dead = precision_score(
        y_true, y_pred_thresh, pos_label=0, zero_division=0
    )
    f1_dead = f1_score(y_true, y_pred_thresh, pos_label=0, zero_division=0)
    f2_dead = _f2_score(precision_dead, recall_dead)

    recall_alive = recall_score(y_true, y_pred_thresh, pos_label=1, zero_division=0)
    precision_alive = precision_score(
        y_true, y_pred_thresh, pos_label=1, zero_division=0
    )
    f1_alive = f1_score(y_true, y_pred_thresh, pos_label=1, zero_division=0)

    cm = confusion_matrix(y_true, y_pred_thresh, labels=[0, 1])
    false_negatives = int(cm[0, 1])

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.0

    try:
        pr_auc = average_precision_score(y_true, y_prob, pos_label=1)
    except ValueError:
        pr_auc = 0.0

    return {
        "model_name": model_name,
        "threshold": round(threshold, 4),
        "accuracy": round(accuracy_score(y_true, y_pred_thresh), 4),
        "recall_dead": round(recall_dead, 4),
        "precision_dead": round(precision_dead, 4),
        "f1_dead": round(f1_dead, 4),
        "f2_dead": round(f2_dead, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "false_negatives": false_negatives,
        "recall_alive": round(recall_alive, 4),
        "precision_alive": round(precision_alive, 4),
        "f1_alive": round(f1_alive, 4),
        "timestamp": datetime.now().isoformat(),
    }


def evaluate_model(
    model_name: str,
    model: Any,
    X_test,
    y_test,
    threshold: float = 0.5,
) -> Dict[str, Union[str, float]]:
    """
    Run get_all_metrics on a fitted model against X_test, y_test.
    Return the metrics dict.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    return get_all_metrics(
        model_name=model_name,
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        threshold=threshold,
    )


def save_results(results: List[Dict], path: str) -> None:
    """
    Accept a list of metric dicts. Convert to DataFrame.
    Save as CSV to path. Print confirmation.
    """
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    print(f"Results saved to {path} ({len(df)} rows)")


def load_results(path: str) -> pd.DataFrame:
    """
    Load a results CSV from path. Return DataFrame.
    """
    return pd.read_csv(path)
