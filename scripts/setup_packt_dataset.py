"""
Fetch Packt Chapter03 SMS Spam dataset, convert to CSV, and preprocess.

Usage:
  python scripts/setup_packt_dataset.py [--force]

Outputs:
  - datasets/raw/SMSSpamCollection
  - datasets/raw/sms_spam_full.csv
  - datasets/processed/sms_spam_clean.csv (+ steps/ and JSON report via preprocess script)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import subprocess

from urllib.request import urlopen

PACKT_RAW_URL = (
    "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/"
    "master/Chapter03/datasets/SMSSpamCollection"
)


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as r, open(dest, "wb") as f:  # nosec - trusted source per user request
        f.write(r.read())


def convert_to_csv(src_txt: Path, out_csv: Path) -> int:
    import pandas as pd

    df = pd.read_csv(src_txt, sep="\t", names=["label", "text"], dtype=str)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return int(len(df))


def run_preprocess(in_csv: Path, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/preprocess_emails.py",
        "--input",
        str(in_csv),
        "--output",
        str(out_csv),
        "--label-col",
        "label",
        "--text-col",
        "text",
        "--output-text-col",
        "text_clean",
        "--save-step-columns",
        "--steps-out-dir",
        "datasets/processed/steps",
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Redownload and overwrite existing files")
    parser.add_argument(
        "--url",
        default=PACKT_RAW_URL,
        help="Source URL for SMSSpamCollection (default: Packt Chapter03 raw)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    raw_txt = repo_root / "datasets/raw/SMSSpamCollection"
    raw_csv = repo_root / "datasets/raw/sms_spam_full.csv"
    processed_csv = repo_root / "datasets/processed/sms_spam_clean.csv"

    if args.force or not raw_txt.exists():
        print(f"Downloading from {args.url} → {raw_txt}")
        download_file(args.url, raw_txt)
    else:
        print(f"Found existing {raw_txt}, skip download (use --force to overwrite)")

    print(f"Converting to CSV → {raw_csv}")
    n_rows = convert_to_csv(raw_txt, raw_csv)
    print(f"Converted rows: {n_rows}")

    print(f"Preprocessing → {processed_csv}")
    run_preprocess(raw_csv, processed_csv)
    print("Done. Set Streamlit 路徑為: datasets/processed/sms_spam_clean.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

