import argparse
from pathlib import Path
from typing import Sequence
import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spam_pipeline import artifacts, metrics as metrics_utils


def parse_range(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("ngram range must be formatted as '1,2'")
    return int(parts[0]), int(parts[1])


def parse_args():
    parser = argparse.ArgumentParser(description="Train a spam classifier with TF-IDF + Logistic Regression.")
    parser.add_argument("--input", required=True, help="Preprocessed CSV containing cleaned text.")
    parser.add_argument("--label-col", default="col_0", help="Name of the label column.")
    parser.add_argument("--text-col", default="text_clean", help="Name of the cleaned text column.")
    parser.add_argument(
        "--numeric-cols",
        nargs="*",
        help="Optional numeric feature columns to include (default: derived *_char_len, *_token_len...).",
    )
    parser.add_argument("--positive-label", default="spam", help="Which label should be treated as positive class.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--models-dir", default="models", help="Directory to store trained artifacts.")
    parser.add_argument("--report-out", default="reports/train_report.txt")
    parser.add_argument("--eval-threshold", type=float, default=0.5)

    # Vectorizer knobs
    parser.add_argument("--ngram-range", type=parse_range, default=(1, 2))
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=3000)
    parser.add_argument("--sublinear-tf", action="store_true")

    # Model knobs
    parser.add_argument("--C", type=float, default=2.0)
    parser.add_argument("--class-weight", choices=["balanced", "auto", "none"], default="balanced")
    parser.add_argument("--max-iter", type=int, default=200)
    return parser.parse_args()


def choose_numeric_cols(df: pd.DataFrame, text_col: str, numeric_override: Sequence[str] | None) -> list[str]:
    if numeric_override:
        missing = [col for col in numeric_override if col not in df.columns]
        if missing:
            raise ValueError(f"Numeric columns not found: {missing}")
        return list(numeric_override)

    prefix = f"{text_col}_"
    return [col for col in df.columns if col.startswith(prefix) and df[col].dtype != "object"]


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    if args.label_col not in df.columns or args.text_col not in df.columns:
        raise ValueError(f"Could not find label/text columns in dataset. Available: {list(df.columns)}")

    numeric_cols = choose_numeric_cols(df, args.text_col, args.numeric_cols)
    y_raw = df[args.label_col].astype(str)
    positive_label = args.positive_label if args.positive_label in set(y_raw) else sorted(set(y_raw))[-1]
    y = (y_raw == positive_label).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    transformers = [
        (
            "text",
            TfidfVectorizer(
                ngram_range=args.ngram_range,
                min_df=args.min_df,
                max_features=args.max_features,
                sublinear_tf=args.sublinear_tf,
            ),
            args.text_col,
        )
    ]
    if numeric_cols:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        sparse_threshold=0.3,
    )

    class_weight = None if args.class_weight == "none" else ("balanced" if args.class_weight == "auto" else args.class_weight)
    clf = LogisticRegression(
        C=args.C,
        class_weight=class_weight,
        max_iter=args.max_iter,
        solver="liblinear",
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", clf),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics_obj, extra = metrics_utils.evaluate_binary_task(y_test, y_proba, threshold=args.eval_threshold)
    y_pred = (y_proba >= args.eval_threshold).astype(int)
    report = classification_report(y_test, y_pred)

    bundle = artifacts.ArtifactBundle(Path(args.models_dir))
    bundle.save(pipeline, "spam_pipeline")
    bundle.save_metadata(
        {
            "label_col": args.label_col,
            "text_col": args.text_col,
            "numeric_cols": numeric_cols,
            "positive_label": positive_label,
            "threshold": args.eval_threshold,
            "vectorizer": {
                "ngram_range": args.ngram_range,
                "min_df": args.min_df,
                "max_features": args.max_features,
                "sublinear_tf": args.sublinear_tf,
            },
            "model": {
                "type": "LogisticRegression",
                "C": args.C,
                "class_weight": class_weight,
                "max_iter": args.max_iter,
            },
            "metrics": metrics_obj.to_dict(),
            "confusion_matrix": extra["confusion_matrix"],
        }
    )

    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fp:
        fp.write("=== Classification Report ===\n")
        fp.write(report)
        fp.write("\n\n=== Metrics ===\n")
        fp.write(str(metrics_obj))
    print(f"Artifacts saved to {args.models_dir}")
    print(f"Evaluation metrics: {metrics_obj}")
    print(f"Classification report saved to {report_path}")


if __name__ == "__main__":
    main()
