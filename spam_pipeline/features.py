from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def derive_text_features(text_series: pd.Series, prefix: str = "text") -> pd.DataFrame:
    """Generate lightweight statistical features from a text column."""
    text_series = text_series.fillna("")
    tokens = text_series.apply(lambda x: TOKEN_RE.findall(x.lower()))

    df = pd.DataFrame(
        {
            f"{prefix}_char_len": text_series.str.len(),
            f"{prefix}_token_len": tokens.apply(len),
            f"{prefix}_digit_count": text_series.str.count(r"\d"),
            f"{prefix}_uppercase_count": text_series.str.count(r"[A-Z]"),
            f"{prefix}_percent_caps": (
                text_series.str.count(r"[A-Z]") / text_series.str.len().clip(lower=1)
            ).fillna(0.0),
            f"{prefix}_avg_token_len": tokens.apply(
                lambda tok: np.mean([len(t) for t in tok]) if tok else 0.0
            ),
            f"{prefix}_bang_count": text_series.str.count("!"),
        }
    )
    return df


def top_tokens_by_class(
    df: pd.DataFrame,
    label_col: str,
    text_col: str,
    topn: int = 20,
) -> dict[str, list[tuple[str, int]]]:
    """Return the most common tokens for each label."""
    results: dict[str, list[tuple[str, int]]] = {}

    def tokenize(text: str) -> list[str]:
        return TOKEN_RE.findall(text.lower())

    for label, group in df.groupby(label_col):
        counts = Counter()
        if group.empty:
            results[str(label)] = []
            continue
        for text in group[text_col].fillna(""):
            counts.update(tokenize(text))
        results[str(label)] = counts.most_common(topn)
    return results


def build_threshold_sweep(
    y_true: Iterable[int],
    y_proba: Iterable[float],
    steps: int = 50,
) -> pd.DataFrame:
    thresholds = np.linspace(0.05, 0.95, steps)
    rows = []
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    for thr in thresholds:
        preds = (y_proba >= thr).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        rows.append({"threshold": thr, "precision": precision, "recall": recall, "f1": f1})
    return pd.DataFrame(rows)
