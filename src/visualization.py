"""Visualization utilities for model comparison and analysis."""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

RANDOM_STATE = 42

STYLE = "seaborn-v0_8-whitegrid"


def _save_or_show(fig: plt.Figure, save_path: Optional[str]) -> None:
    """Save figure to path or display it."""
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _get_model_category(model_name: str) -> str:
    """Assign a category for coloring based on model name."""
    name_lower = model_name.lower()
    if any(
        kw in name_lower
        for kw in ["smote", "adasyn", "none", "rf +"]
    ):
        return "data-level"
    if any(
        kw in name_lower
        for kw in ["balanced", "easy", "xgboost", "lightgbm", "xgb", "lgbm"]
    ):
        return "algorithm-level"
    return "baseline"


def _category_color(category: str) -> str:
    """Return color for a model category."""
    return {
        "data-level": "steelblue",
        "algorithm-level": "purple",
        "baseline": "gray",
    }.get(category, "gray")


def plot_class_distribution(y, save_path: Optional[str] = None) -> None:
    """
    Bar chart: Alive vs Dead counts with percentage annotations.
    Two bars (blue=Alive, red=Dead).
    Add a horizontal dashed line at the 50% mark labelled 'balanced'.
    Title: 'Class distribution — Alive vs Dead'.
    Figure size: (8, 5).
    """
    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(8, 5))

    counts = pd.Series(y).value_counts().sort_index()
    labels = ["Dead", "Alive"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    total = sum(values)
    colors = ["red", "blue"]

    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, values):
        pct = 100 * val / total if total > 0 else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.01,
            f"{val}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    max_count = max(values) if values else 1
    balanced_line = 0.5 * total
    ax.axhline(
        balanced_line,
        color="green",
        linestyle="--",
        linewidth=1.5,
        label="balanced (50%)",
    )
    ax.set_ylim(0, max_count * 1.15)
    ax.set_ylabel("Count")
    ax.set_title("Class distribution — Alive vs Dead")
    ax.legend()

    _save_or_show(fig, save_path)


def plot_pr_curves(
    results_list: List[Dict],
    save_path: Optional[str] = None,
) -> None:
    """
    Accept a list of dicts, each with keys:
      model_name, y_true, y_prob
    Plot one PR curve per model, all overlaid.
    Each curve labelled '{model_name} (AP={ap:.3f})'.
    Add a horizontal dashed line at the class prevalence (baseline).
    Title: 'Precision-Recall curves — all models'.
    Legend outside right. Figure size: (10, 6).
    """
    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(10, 6))

    prevalence = None
    for entry in results_list:
        y_true = np.asarray(entry["y_true"])
        y_prob = np.asarray(entry["y_prob"])
        model_name = entry["model_name"]

        if prevalence is None:
            prevalence = y_true.mean()

        precision, recall, _ = precision_recall_curve(y_true, y_prob, pos_label=1)
        ap = average_precision_score(y_true, y_prob, pos_label=1)
        ax.plot(
            recall,
            precision,
            label=f"{model_name} (AP={ap:.3f})",
            linewidth=1.5,
        )

    if prevalence is not None:
        ax.axhline(
            prevalence,
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"baseline ({prevalence:.3f})",
        )

    ax.set_xlabel("Recall (Alive)")
    ax.set_ylabel("Precision (Alive)")
    ax.set_title("Precision-Recall curves — all models")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    _save_or_show(fig, save_path)


def plot_false_negatives_bar(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """
    Horizontal bar chart.
    One bar per model, sorted by false_negatives ascending.
    Color bars by category:
      data-level sampling = blue, algorithm-level = purple, baseline = gray.
    Annotate each bar with its false_negatives count.
    Add a vertical dashed line at the baseline value.
    Title: 'False negatives (Dead patients missed) — lower is better'.
    Figure size: (10, 7).
    """
    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(10, 7))

    df = results_df.copy()
    df["category"] = df["model_name"].apply(_get_model_category)
    df = df.sort_values("false_negatives", ascending=True)

    baseline_rows = df[df["category"] == "baseline"]
    baseline_fn = (
        baseline_rows["false_negatives"].iloc[0]
        if len(baseline_rows) > 0
        else df["false_negatives"].max()
    )

    colors = [_category_color(c) for c in df["category"]]
    bars = ax.barh(
        df["model_name"],
        df["false_negatives"],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )

    for bar, fn in zip(bars, df["false_negatives"]):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            str(int(fn)),
            va="center",
            fontsize=9,
        )

    ax.axvline(
        baseline_fn,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"baseline ({int(baseline_fn)})",
    )
    ax.set_xlabel("False negatives (Dead patients missed)")
    ax.set_title("False negatives (Dead patients missed) — lower is better")
    ax.legend()

    _save_or_show(fig, save_path)


def plot_recall_precision_scatter(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """
    Scatter plot: x=recall_dead, y=precision_dead, one labelled dot per model.
    Color by category (same as false negatives bar).
    Add F1=0.4 iso-curve as a dashed line for reference.
    Draw dotted lines at x=0.5 and y=0.25 to create quadrants.
    Annotate each point with model_name.
    Title: 'Recall vs Precision trade-off — Dead class'.
    Figure size: (10, 7).
    """
    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(10, 7))

    df = results_df.copy()
    df["category"] = df["model_name"].apply(_get_model_category)

    recalls = np.linspace(0.01, 1, 200)
    f1_target = 0.4
    precisions_iso = (f1_target * recalls) / (2 * recalls - f1_target)
    valid = (2 * recalls - f1_target) > 0
    ax.plot(
        recalls[valid],
        precisions_iso[valid],
        "k--",
        linewidth=1,
        alpha=0.6,
        label="F1 = 0.4",
    )

    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.axhline(0.25, color="gray", linestyle=":", linewidth=1)

    for _, row in df.iterrows():
        color = _category_color(row["category"])
        ax.scatter(
            row["recall_dead"],
            row["precision_dead"],
            c=color,
            s=80,
            edgecolors="black",
            linewidth=0.5,
            zorder=3,
        )
        ax.annotate(
            row["model_name"],
            (row["recall_dead"], row["precision_dead"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    ax.set_xlabel("Recall (Dead)")
    ax.set_ylabel("Precision (Dead)")
    ax.set_title("Recall vs Precision trade-off — Dead class")
    ax.legend()
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    _save_or_show(fig, save_path)


def plot_threshold_sensitivity(
    model_name: str,
    y_true,
    y_prob,
    save_path: Optional[str] = None,
) -> None:
    """
    Line chart: x=threshold (0.05 to 0.60 in steps of 0.01),
    three lines: recall_dead, precision_dead, f2_dead.
    Annotate the default threshold (0.5) with a vertical dashed line.
    Find and annotate the F2-optimal threshold with a vertical line.
    Title: f'Threshold sensitivity — {model_name}'.
    Figure size: (10, 5).
    """
    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(10, 5))

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    thresholds = np.arange(0.05, 0.61, 0.01)

    recalls, precisions, f2s = [], [], []
    for t in thresholds:
        p_dead = 1 - y_prob
        y_pred = np.where(p_dead > t, 0, 1)
        rec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        prec = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
        f2 = (5 * prec * rec) / (4 * prec + rec) if (prec + rec) > 0 else 0
        recalls.append(rec)
        precisions.append(prec)
        f2s.append(f2)

    ax.plot(thresholds, recalls, label="Recall (Dead)", linewidth=2)
    ax.plot(thresholds, precisions, label="Precision (Dead)", linewidth=2)
    ax.plot(thresholds, f2s, label="F2 (Dead)", linewidth=2)

    ax.axvline(
        0.5,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label="default (0.5)",
    )

    best_idx = int(np.argmax(f2s))
    best_threshold = thresholds[best_idx]
    ax.axvline(
        best_threshold,
        color="green",
        linestyle="--",
        linewidth=1.5,
        label=f"F2-optimal ({best_threshold:.2f})",
    )

    ax.set_xlabel("Threshold on P(Dead)")
    ax.set_ylabel("Score")
    ax.set_title(f"Threshold sensitivity — {model_name}")
    ax.legend()
    ax.set_xlim(0.05, 0.60)
    ax.set_ylim(0, 1.05)

    _save_or_show(fig, save_path)


def plot_master_comparison(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """
    Grouped bar chart comparing all models across four key metrics:
    recall_dead, pr_auc, roc_auc, false_negatives (normalised 0-1).
    Each model is a group of four bars.
    Add a table of exact values below the chart.
    Title: 'Master comparison — all models and metrics'.
    Figure size: (14, 7).
    """
    plt.style.use(STYLE)

    metrics = ["recall_dead", "pr_auc", "roc_auc", "false_negatives"]
    df = results_df.copy().sort_values("recall_dead", ascending=False)

    norm_data = df[metrics].copy()
    fn_max = norm_data["false_negatives"].max()
    if fn_max > 0:
        norm_data["false_negatives"] = 1 - (
            norm_data["false_negatives"] / fn_max
        )
    else:
        norm_data["false_negatives"] = 1.0

    n_models = len(df)
    n_metrics = len(metrics)
    x = np.arange(n_models)
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 7))
    metric_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, metric in enumerate(metrics):
        offset = (i - n_metrics / 2 + 0.5) * width
        ax.bar(
            x + offset,
            norm_data[metric],
            width,
            label=metric,
            color=metric_colors[i],
            edgecolor="black",
            linewidth=0.3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df["model_name"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Normalised score (0–1)")
    ax.set_title("Master comparison — all models and metrics")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.15)

    table_data = []
    for _, row in df.iterrows():
        table_data.append(
            [
                f"{row['recall_dead']:.3f}",
                f"{row['pr_auc']:.3f}",
                f"{row['roc_auc']:.3f}",
                str(int(row["false_negatives"])),
            ]
        )

    table = ax.table(
        cellText=table_data,
        colLabels=["recall_dead", "pr_auc", "roc_auc", "FN"],
        rowLabels=df["model_name"].tolist(),
        loc="bottom",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.2)

    plt.subplots_adjust(bottom=0.35)
    _save_or_show(fig, save_path)
