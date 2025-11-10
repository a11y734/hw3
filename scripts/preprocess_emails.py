import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spam_pipeline import features, io_utils, preprocessing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess SMS/Email spam datasets with configurable cleaning steps.",
    )
    parser.add_argument("--input", required=True, help="Path to the raw CSV dataset.")
    parser.add_argument("--output", required=True, help="Destination CSV for the cleaned dataset.")
    parser.add_argument(
        "--metadata-out",
        default="datasets/processed/preprocess_report.json",
        help="JSON file summarizing preprocessing statistics.",
    )
    parser.add_argument("--label-col", help="Name of the label column.")
    parser.add_argument("--text-col", help="Name of the text/message column.")
    parser.add_argument(
        "--label-col-index",
        type=int,
        default=0,
        help="Label column index when --label-col is not provided.",
    )
    parser.add_argument(
        "--text-col-index",
        type=int,
        default=1,
        help="Text column index when --text-col is not provided.",
    )
    parser.add_argument("--no-header", action="store_true", help="Set if the CSV has no header row.")
    parser.add_argument(
        "--output-text-col",
        default="text_clean",
        help="Name of the cleaned text column to append to the dataset.",
    )
    parser.add_argument("--keep-duplicates", action="store_true", help="Skip duplicate removal.")
    parser.add_argument(
        "--save-step-columns",
        action="store_true",
        help="Save every preprocessing step as a CSV column under --steps-out-dir.",
    )
    parser.add_argument(
        "--steps-out-dir",
        default="datasets/processed/steps",
        help="Folder to store per-step CSVs when --save-step-columns is used.",
    )
    parser.add_argument("--encoding", default="utf-8", help="File encoding.")

    # Cleaning toggles
    parser.add_argument("--keep-case", action="store_true", help="Disable lowercase conversion.")
    parser.add_argument("--keep-html", action="store_true", help="Disable HTML stripping.")
    parser.add_argument("--keep-urls", action="store_true", help="Disable URL removal.")
    parser.add_argument("--keep-emails", action="store_true", help="Disable e-mail removal.")
    parser.add_argument("--keep-punct", action="store_true", help="Disable punctuation removal.")
    parser.add_argument(
        "--remove-numbers",
        action="store_true",
        help="Remove number sequences from the text.",
    )
    parser.add_argument(
        "--stopwords",
        nargs="*",
        default=None,
        help="Optional custom stopwords (space separated).",
    )
    parser.add_argument(
        "--remove-stopwords",
        action="store_true",
        help="Remove English stopwords (plus any custom ones).",
    )
    parser.add_argument("--stem", action="store_true", help="Apply Snowball stemming.")
    parser.add_argument("--stem-language", default="english")
    return parser.parse_args()


def pick_column(df: pd.DataFrame, name: str | None, index: int) -> str:
    if name and name in df.columns:
        return name
    if index >= len(df.columns):
        raise ValueError(f"Column index {index} is out of bounds for dataset columns {list(df.columns)}")
    return df.columns[index]


def main():
    args = parse_args()
    dataset = io_utils.load_csv_dataset(
        args.input,
        has_header=not args.no_header,
        encoding=args.encoding,
    )

    label_col = pick_column(dataset, args.label_col, args.label_col_index)
    text_col = pick_column(dataset, args.text_col, args.text_col_index)

    df = dataset.copy()
    original_count = len(df)
    df = preprocessing.drop_unlabeled(df, label_col, text_col)
    if not args.keep_duplicates:
        df = preprocessing.deduplicate_examples(df, label_col, text_col)

    config = preprocessing.PreprocessConfig(
        lowercase=not args.keep_case,
        strip_html=not args.keep_html,
        remove_urls=not args.keep_urls,
        remove_emails=not args.keep_emails,
        remove_numbers=args.remove_numbers,
        remove_punctuation=not args.keep_punct,
        remove_stopwords=args.remove_stopwords,
        custom_stopwords=args.stopwords,
        stemming=args.stem,
        stem_language=args.stem_language,
    )

    cleaned, steps = preprocessing.preprocess_text_column(df[text_col], config)
    df[args.output_text_col] = cleaned

    feature_df = features.derive_text_features(cleaned, prefix=args.output_text_col)
    df = pd.concat([df.reset_index(drop=True), feature_df], axis=1)

    if args.save_step_columns:
        step_paths = io_utils.save_step_outputs(steps, Path(args.steps_out_dir))
    else:
        step_paths = []

    output_path = io_utils.save_dataframe(df, Path(args.output))
    metadata = {
        "input_rows": original_count,
        "output_rows": len(df),
        "label_col": label_col,
        "text_col": text_col,
        "clean_text_col": args.output_text_col,
        "config": config.to_dict(),
        "step_files": [str(p) for p in step_paths],
        "column_candidates": io_utils.detect_column_candidates(dataset),
        "step_summary": preprocessing.summarize_steps(steps),
    }
    io_utils.write_json(metadata, Path(args.metadata_out))

    print(f"Saved cleaned dataset to {output_path}")
    if step_paths:
        print(f"Stored {len(step_paths)} step outputs under {args.steps_out_dir}")
    print(f"Metadata: {args.metadata_out}")


if __name__ == "__main__":
    main()
