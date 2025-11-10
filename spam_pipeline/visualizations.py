from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

sns.set_theme(style="whitegrid")


def save_figure(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_class_distribution(df, label_col: str):
    counts = df[label_col].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    data = counts.reset_index()
    data.columns = ["label", "count"]
    sns.barplot(data=data, x="label", y="count", hue="label", ax=ax, palette="viridis", legend=False)
    ax.set_title("Class Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    max_count = data["count"].max()
    for idx, value in enumerate(data["count"]):
        ax.text(idx, value + max_count * 0.01, f"{value}", ha="center")
    return fig


def plot_token_frequency(top_tokens: dict[str, list[tuple[str, int]]], label_order=None):
    if label_order is None:
        label_order = list(top_tokens.keys())
    rows = []
    for label in label_order:
        for token, count in top_tokens.get(label, []):
            rows.append({"label": label, "token": token, "count": count})
    if not rows:
        raise ValueError("No token statistics to visualize.")
    import pandas as pd

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(
        data=df,
        y="token",
        x="count",
        hue="label",
        dodge=True,
        ax=ax,
    )
    ax.set_title("Top Tokens by Class")
    return fig


def plot_confusion_matrix(cm: Iterable[Iterable[int]], labels: list[str]):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    return fig


def plot_roc_curve(y_true, y_score):
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    auc = metrics.auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    return fig


def plot_precision_recall(y_true, y_score):
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
    auc = metrics.auc(recall, precision)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(recall, precision, label=f"AUC={auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    return fig


def plot_threshold_sweep(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["threshold"], df["precision"], label="Precision")
    ax.plot(df["threshold"], df["recall"], label="Recall")
    ax.plot(df["threshold"], df["f1"], label="F1")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Sweep")
    ax.legend()
    return fig
