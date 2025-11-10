from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn import metrics


@dataclass
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    threshold: float

    def to_dict(self) -> dict:
        return self.__dict__


def evaluate_binary_task(
    y_true: Iterable[int],
    y_proba: Iterable[float],
    threshold: float = 0.5,
) -> tuple[BinaryMetrics, dict]:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    preds = (y_proba >= threshold).astype(int)

    accuracy = metrics.accuracy_score(y_true, preds)
    precision = metrics.precision_score(y_true, preds, zero_division=0)
    recall = metrics.recall_score(y_true, preds, zero_division=0)
    f1 = metrics.f1_score(y_true, preds, zero_division=0)
    roc_auc = metrics.roc_auc_score(y_true, y_proba)
    precision_curve, recall_curve, _ = metrics.precision_recall_curve(y_true, y_proba)
    pr_auc = metrics.auc(recall_curve, precision_curve)
    cm = metrics.confusion_matrix(y_true, preds).tolist()

    return (
        BinaryMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            threshold=threshold,
        ),
        {"confusion_matrix": cm},
    )


def classification_report(y_true: Iterable[int], y_pred: Iterable[int]) -> str:
    return metrics.classification_report(y_true, y_pred, digits=3)
