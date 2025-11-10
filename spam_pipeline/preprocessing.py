from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable, Optional

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
HTML_RE = re.compile(r"<[^>]+>")
PUNCT_RE = re.compile(r"[^\w\s']")
MULTI_WS_RE = re.compile(r"\s+")


@dataclass
class PreprocessConfig:
    lowercase: bool = True
    strip_html: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_numbers: bool = False
    remove_punctuation: bool = True
    collapse_whitespace: bool = True
    remove_stopwords: bool = False
    custom_stopwords: Optional[Iterable[str]] = None
    stemming: bool = False
    stem_language: str = "english"

    def to_dict(self) -> dict:
        cfg = asdict(self)
        if self.custom_stopwords is not None:
            cfg["custom_stopwords"] = list(self.custom_stopwords)
        return cfg


def _apply_stopwords(series: pd.Series, custom: Optional[Iterable[str]] = None) -> pd.Series:
    stopwords = set(ENGLISH_STOP_WORDS)
    if custom:
        stopwords.update({w.lower() for w in custom})

    def remove_words(text: str) -> str:
        tokens = text.split()
        kept = [tok for tok in tokens if tok.lower() not in stopwords]
        return " ".join(kept)

    return series.apply(remove_words)


def _apply_stemming(series: pd.Series, language: str) -> pd.Series:
    try:
        from nltk.stem import SnowballStemmer
    except ImportError as exc:
        raise RuntimeError(
            "nltk is required for stemming. Install it or disable --stem."
        ) from exc

    stemmer = SnowballStemmer(language)
    return series.apply(lambda text: " ".join(stemmer.stem(token) for token in text.split()))


def preprocess_text_column(
    text_series: pd.Series,
    config: PreprocessConfig,
) -> tuple[pd.Series, list[tuple[str, pd.Series]]]:
    """Clean a text column and optionally capture intermediate steps."""
    steps: list[tuple[str, pd.Series]] = []
    current = text_series.fillna("").astype(str)
    steps.append(("00_original", current))

    if config.lowercase:
        current = current.str.lower()
        steps.append(("01_lowercase", current))

    if config.strip_html:
        current = current.apply(lambda x: HTML_RE.sub(" ", x))
        steps.append(("02_strip_html", current))

    if config.remove_urls:
        current = current.apply(lambda x: URL_RE.sub(" ", x))
        steps.append(("03_remove_urls", current))

    if config.remove_emails:
        current = current.apply(lambda x: EMAIL_RE.sub(" ", x))
        steps.append(("04_remove_emails", current))

    if config.remove_numbers:
        current = current.str.replace(r"\d+", " ", regex=True)
        steps.append(("05_remove_numbers", current))

    if config.remove_punctuation:
        current = current.apply(lambda x: PUNCT_RE.sub(" ", x))
        steps.append(("06_remove_punctuation", current))

    if config.collapse_whitespace:
        current = current.apply(lambda x: MULTI_WS_RE.sub(" ", x).strip())
        steps.append(("07_collapse_ws", current))

    if config.remove_stopwords:
        current = _apply_stopwords(current, config.custom_stopwords)
        steps.append(("08_remove_stopwords", current))

    if config.stemming:
        current = _apply_stemming(current, config.stem_language)
        steps.append(("09_stem", current))

    return current, steps


def deduplicate_examples(df: pd.DataFrame, label_col: str, text_col: str) -> pd.DataFrame:
    """Remove duplicated rows keeping the first occurrence."""
    return df.drop_duplicates(subset=[label_col, text_col])


def drop_unlabeled(df: pd.DataFrame, label_col: str, text_col: str) -> pd.DataFrame:
    return df.dropna(subset=[label_col, text_col])


def summarize_steps(steps: list[tuple[str, pd.Series]]) -> list[dict]:
    return [
        {
            "name": name,
            "non_null": int(series.replace("", pd.NA).dropna().shape[0]),
            "avg_length": float(series.str.len().mean()),
        }
        for name, series in steps
    ]
