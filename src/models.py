"""Model definitions and training utilities."""

from typing import Any, Dict, Optional

from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "vanilla_rf": {
        "class": RandomForestClassifier,
        "params": {
            "n_estimators": 1000,
            "max_depth": 30,
            "max_features": "log2",
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        },
    },
    "balanced_rf": {
        "class": BalancedRandomForestClassifier,
        "params": {
            "n_estimators": 1000,
            "max_features": "log2",
            "max_depth": 30,
            "sampling_strategy": "auto",
            "replacement": False,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        },
    },
    "easy_ensemble": {
        "class": EasyEnsembleClassifier,
        "params": {
            "n_estimators": 50,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        },
    },
    "xgboost": {
        "class": XGBClassifier,
        "params": {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "eval_metric": "aucpr",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": 0,
        },
    },
    "lightgbm": {
        "class": LGBMClassifier,
        "params": {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "is_unbalance": True,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbose": -1,
        },
    },
}


def build_model(
    model_type: str,
    class_weight: Optional[str] = None,
    scale_pos_weight: Optional[float] = None,
) -> Any:
    """
    Build and return an unfitted model from MODEL_CONFIGS.
    If class_weight is provided and the model supports it, add it to params.
    If scale_pos_weight is provided and model_type is 'xgboost', add it.
    Raise ValueError for unknown model_type.
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Choose from: {list(MODEL_CONFIGS.keys())}"
        )

    config = MODEL_CONFIGS[model_type]
    params = config["params"].copy()

    if class_weight is not None and model_type in ("vanilla_rf",):
        params["class_weight"] = class_weight

    if scale_pos_weight is not None and model_type == "xgboost":
        params["scale_pos_weight"] = scale_pos_weight

    return config["class"](**params)


def train_model(model: Any, X_train, y_train) -> Any:
    """
    Fit model on X_train, y_train. Return fitted model.
    """
    model.fit(X_train, y_train)
    return model
