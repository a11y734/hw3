import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spam_pipeline import artifacts, features, metrics as metrics_utils, visualizations


def parse_args():
    parser = argparse.ArgumentParser(description="Generate visual reports for the spam dataset and model.")
    parser.add_argument("--input", required=True, help="CSV file with data.")
    parser.add_argument("--label-col", default="col_0")
    parser.add_argument("--text-col", default="text_clean")
    parser.add_argument("--reports-dir", default="reports/visualizations")
    parser.add_argument("--class-dist", action="store_true")
    parser.add_argument("--token-freq", action="store_true")
    parser.add_argument("--topn", type=int, default=20)
    parser.add_argument("--models-dir", default="models", help="Load trained model for evaluation plots.")
    parser.add_argument("--confusion-matrix", action="store_true")
    parser.add_argument("--roc", action="store_true")
    parser.add_argument("--pr", action="store_true")
    parser.add_argument("--threshold-sweep", action="store_true")
    parser.add_argument("--threshold", type=float, help="Override threshold for CM display.")
    parser.add_argument("--show", action="store_true", help="Display figures interactively.")
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if args.class_dist:
        fig = visualizations.plot_class_distribution(df, args.label_col)
        if args.show:
            fig.show()
        path = visualizations.save_figure(fig, reports_dir / "class_distribution.png")
        print(f"Class distribution saved to {path}")

    if args.token_freq:
        top_tokens = features.top_tokens_by_class(df, args.label_col, args.text_col, args.topn)
        fig = visualizations.plot_token_frequency(top_tokens)
        if args.show:
            fig.show()
        path = visualizations.save_figure(fig, reports_dir / "token_frequency.png")
        print(f"Token frequency saved to {path}")

    needs_model = any([args.confusion_matrix, args.roc, args.pr, args.threshold_sweep])
    if needs_model:
        bundle = artifacts.ArtifactBundle(Path(args.models_dir))
        pipeline = bundle.load("spam_pipeline")
        metadata = bundle.load_metadata()
        positive_label = metadata.get("positive_label", "spam")
        threshold = args.threshold or metadata.get("metrics", {}).get("threshold", 0.5)

        y_true = (df[args.label_col].astype(str) == positive_label).astype(int)
        y_proba = pipeline.predict_proba(df)[:, 1]

        if args.confusion_matrix:
            metrics_obj, extra = metrics_utils.evaluate_binary_task(y_true, y_proba, threshold)
            fig = visualizations.plot_confusion_matrix(extra["confusion_matrix"], [f"not_{positive_label}", positive_label])
            if args.show:
                fig.show()
            path = visualizations.save_figure(fig, reports_dir / "confusion_matrix.png")
            print(f"Confusion matrix saved to {path} @ threshold {threshold:.2f}")

        if args.roc:
            fig = visualizations.plot_roc_curve(y_true, y_proba)
            if args.show:
                fig.show()
            path = visualizations.save_figure(fig, reports_dir / "roc_curve.png")
            print(f"ROC curve saved to {path}")

        if args.pr:
            fig = visualizations.plot_precision_recall(y_true, y_proba)
            if args.show:
                fig.show()
            path = visualizations.save_figure(fig, reports_dir / "precision_recall.png")
            print(f"PR curve saved to {path}")

        if args.threshold_sweep:
            sweep_df = features.build_threshold_sweep(y_true, y_proba)
            csv_path = reports_dir / "threshold_sweep.csv"
            sweep_df.to_csv(csv_path, index=False)
            fig = visualizations.plot_threshold_sweep(sweep_df)
            if args.show:
                fig.show()
            img_path = visualizations.save_figure(fig, reports_dir / "threshold_sweep.png")
            print(f"Threshold sweep saved to {img_path} and data to {csv_path}")


if __name__ == "__main__":
    main()
