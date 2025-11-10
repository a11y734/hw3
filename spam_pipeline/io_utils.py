import json
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def ensure_dir(path: Path) -> Path:
    """Create parent directories as needed and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_csv_dataset(
    csv_path: Path,
    has_header: bool = True,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Load the SMS spam dataset, auto-assigning column names when no header exists."""
    csv_path = Path(csv_path)
    read_kwargs = {"encoding": encoding}
    if has_header:
        return pd.read_csv(csv_path, **read_kwargs)

    df = pd.read_csv(csv_path, header=None, **read_kwargs)
    df.columns = [f"col_{idx}" for idx in range(df.shape[1])]
    return df


def save_dataframe(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    path = ensure_dir(Path(path))
    df.to_csv(path, index=index)
    return path


def write_json(data: dict, path: Path) -> Path:
    path = ensure_dir(Path(path))
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)
    return path


def detect_column_candidates(df: pd.DataFrame, max_preview: int = 5) -> dict:
    """Return a quick view of potential label/text columns for CLI hints."""
    preview = df.head(max_preview).to_dict(orient="list")
    return {
        "columns": list(df.columns),
        "preview_rows": max_preview,
        "examples": preview,
    }


def save_step_outputs(
    steps: Iterable[tuple[str, pd.Series]],
    target_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for order, (name, series) in enumerate(steps):
        out_path = target_dir / f"{order:02d}_{name}.csv"
        series.to_csv(out_path, header=[name], index=False)
        paths.append(out_path)
    return paths
