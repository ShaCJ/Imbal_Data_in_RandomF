"""Sampling strategies for handling class imbalance."""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.decomposition import PCA

RANDOM_STATE = 42

SAMPLER_MAP = {
    "none": None,
    "smote": SMOTE(random_state=RANDOM_STATE),
    "adasyn": ADASYN(random_state=RANDOM_STATE),
    "smoteenn": SMOTEENN(random_state=RANDOM_STATE),
    "smotetomek": SMOTETomek(random_state=RANDOM_STATE),
}


def apply_sampler(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = "smote",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply the named sampler to X_train, y_train.
    method must be one of: 'none', 'smote', 'adasyn', 'smoteenn',
    'smotetomek'.
    If method is 'none', return X_train, y_train unchanged.
    Returns X_resampled, y_resampled.
    Print a summary: original shape, resampled shape, class counts.
    """
    if method not in SAMPLER_MAP:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Choose from: {list(SAMPLER_MAP.keys())}"
        )

    original_counts = y_train.value_counts().to_dict()
    print(f"Original shape: {X_train.shape}")
    print(f"Original class counts: {original_counts}")

    sampler = SAMPLER_MAP[method]
    if sampler is None:
        X_resampled, y_resampled = X_train.copy(), y_train.copy()
    else:
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        if not isinstance(X_resampled, pd.DataFrame):
            X_resampled = pd.DataFrame(
                X_resampled, columns=X_train.columns
            )
        if not isinstance(y_resampled, pd.Series):
            y_resampled = pd.Series(y_resampled, name=y_train.name)

    resampled_counts = pd.Series(y_resampled).value_counts().to_dict()
    print(f"Resampled shape: {X_resampled.shape}")
    print(f"Resampled class counts: {resampled_counts}")

    return X_resampled, y_resampled


def plot_sampling_comparison(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    methods: List[str],
    save_path: Optional[str] = None,
) -> None:
    """
    Create a 2x2 PCA scatter grid comparing the training data before
    and after sampling for each method in methods list.
    Majority class = blue, minority class = red, synthetic = green.
    Each panel titled with the method name.
    If save_path is provided, save as PNG at 150 DPI.
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    n_methods = len(methods)
    n_cols = 2
    n_rows = int(np.ceil(n_methods / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    orig_rows = {
        tuple(row) for row in np.round(X_train.values.astype(float), 6)
    }

    for ax_idx, method in enumerate(methods):
        ax = axes[ax_idx]

        if method == "none":
            X_plot, y_plot = X_train.copy(), y_train.copy()
        else:
            X_plot, y_plot = apply_sampler(X_train, y_train, method=method)

        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        X_pca = pca.fit_transform(X_plot)

        y_arr = np.asarray(y_plot)
        rounded_rows = np.round(
            np.asarray(X_plot, dtype=float), 6
        )
        if method != "none":
            synthetic_mask = np.array(
                [
                    tuple(rounded_rows[i]) not in orig_rows
                    for i in range(len(y_arr))
                ]
            )
            original_mask = ~synthetic_mask
        else:
            synthetic_mask = np.zeros(len(y_arr), dtype=bool)
            original_mask = np.ones(len(y_arr), dtype=bool)

        majority_mask = (y_arr == 1) & original_mask
        minority_mask = (y_arr == 0) & original_mask

        ax.scatter(
            X_pca[majority_mask, 0],
            X_pca[majority_mask, 1],
            c="blue",
            alpha=0.5,
            s=15,
            label="Alive (majority)",
        )
        ax.scatter(
            X_pca[minority_mask, 0],
            X_pca[minority_mask, 1],
            c="red",
            alpha=0.5,
            s=15,
            label="Dead (minority)",
        )

        if method != "none" and synthetic_mask.any():
            ax.scatter(
                X_pca[synthetic_mask, 0],
                X_pca[synthetic_mask, 1],
                c="green",
                alpha=0.5,
                s=15,
                label="Synthetic",
            )

        ax.set_title(method.upper())
        ax.legend(loc="best", fontsize=8)

    for ax in axes[n_methods:]:
        ax.set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
