import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spam_pipeline import artifacts, features


def parse_args():
    parser = argparse.ArgumentParser(description="Predict spam/ham labels using trained artifacts.")
    parser.add_argument("--models-dir", default="models", help="Directory containing model artifacts.")
    parser.add_argument("--text", help="Single text snippet to classify.")
    parser.add_argument("--input", help="CSV file for batch inference.")
    parser.add_argument("--text-col", help="Text column name for batch inference.")
    parser.add_argument("--output", help="Optional CSV path to save predictions for batch mode.")
    parser.add_argument("--threshold", type=float, help="Override probability threshold.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.text and not args.input:
        raise SystemExit("Provide either --text or --input for predictions.")

    bundle = artifacts.ArtifactBundle(Path(args.models_dir))
    pipeline = bundle.load("spam_pipeline")
    metadata = bundle.load_metadata()
    positive_label = metadata.get("positive_label", "spam")
    threshold = args.threshold or metadata.get("metrics", {}).get("threshold", 0.5)

    text_col = metadata.get("text_col", "text_clean")
    numeric_cols = metadata.get("numeric_cols", [])

    if args.text:
        df = build_inference_frame([args.text], text_col, numeric_cols)
        probs = pipeline.predict_proba(df)[:, 1]
        label = positive_label if probs[0] >= threshold else f"not_{positive_label}"
        print(f"Text: {args.text}")
        print(f"Spam probability: {probs[0]:.4f}")
        print(f"Prediction ({threshold:.2f}): {label}")
        return

    if args.text_col:
        text_col = args.text_col

    batch_df = pd.read_csv(args.input)
    if text_col not in batch_df.columns:
        raise SystemExit(f"Column '{text_col}' not found in {args.input}.")

    probs = pipeline.predict_proba(batch_df)[:, 1]
    preds = [positive_label if p >= threshold else f"not_{positive_label}" for p in probs]
    result = batch_df.copy()
    result["spam_probability"] = probs
    result["prediction"] = preds

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"Saved predictions to {output_path}")
    else:
        print(result[["spam_probability", "prediction"]].head())


def build_inference_frame(texts: list[str], text_col: str, numeric_cols: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({text_col: texts})
    if numeric_cols:
        derived = features.derive_text_features(df[text_col], prefix=text_col)
        for col in numeric_cols:
            if col not in derived.columns:
                derived[col] = 0.0
        df = pd.concat([df, derived[numeric_cols]], axis=1)
    return df


if __name__ == "__main__":
    main()
