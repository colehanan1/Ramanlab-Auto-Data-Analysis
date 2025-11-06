"""Invoke the flybehavior-response CLI to score reactions."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Sequence, Set, Tuple

import pandas as pd

from ..config import Settings


NON_REACTIVE_SPAN_PX = 20.0


def _stringify(value: object, fallback: str = "UNKNOWN") -> str:
    """Return a normalised string representation for fly identifiers."""

    if isinstance(value, str):
        text = value.strip()
        return text or fallback
    if value is None:
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except TypeError:
        pass
    text = str(value).strip()
    return text or fallback


def _string_series(df: pd.DataFrame, column: str, *, fallback: str = "UNKNOWN") -> pd.Series:
    if column not in df.columns:
        return pd.Series([fallback] * len(df), index=df.index, dtype=object)
    series = df[column]
    return series.apply(lambda value: _stringify(value, fallback))


def _non_reactive_mask(
    df: pd.DataFrame, *, threshold: float = NON_REACTIVE_SPAN_PX
) -> pd.Series:
    if {"trimmed_global_min", "trimmed_global_max"}.issubset(df.columns):
        gmin = pd.to_numeric(df["trimmed_global_min"], errors="coerce")
        gmax = pd.to_numeric(df["trimmed_global_max"], errors="coerce")
        base_min = pd.to_numeric(df.get("global_min"), errors="coerce")
        base_max = pd.to_numeric(df.get("global_max"), errors="coerce")
        gmin = gmin.where(gmin.notna(), base_min)
        gmax = gmax.where(gmax.notna(), base_max)
    else:
        gmin = pd.to_numeric(df.get("global_min"), errors="coerce")
        gmax = pd.to_numeric(df.get("global_max"), errors="coerce")
    if gmin is None or gmax is None:
        return pd.Series(False, index=df.index, dtype=bool)
    span = (gmax - gmin).abs()
    mask = gmin.notna() & gmax.notna() & span.le(float(threshold))
    return mask.fillna(False)


def _fly_keys(df: pd.DataFrame) -> pd.Series:
    dataset = _string_series(df, "dataset")
    fly = _string_series(df, "fly")
    fly_number = _string_series(df, "fly_number")
    return pd.Series(list(zip(dataset, fly, fly_number)), index=df.index, dtype=object)


def _drop_flagged_flies(
    df: pd.DataFrame, *, threshold: float = NON_REACTIVE_SPAN_PX
) -> tuple[pd.DataFrame, Set[Tuple[str, str, str]]]:
    mask = _non_reactive_mask(df, threshold=threshold)
    keys = _fly_keys(df)
    flagged_pairs = set(keys[mask])
    if not flagged_pairs:
        return df.copy(), set()

    keep_mask = ~keys.isin(flagged_pairs)
    filtered = df.loc[keep_mask].copy()
    return filtered, flagged_pairs


def _filter_trial_types(
    df: pd.DataFrame, allowed: Iterable[str] = ("testing",)
) -> pd.DataFrame:
    """Return only the rows whose ``trial_type`` matches ``allowed``."""

    if "trial_type" not in df.columns:
        return df.copy()

    allowed_normalised = {str(value).strip().lower() for value in allowed}
    if not allowed_normalised:
        return df.copy()

    mask = (
        df["trial_type"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(allowed_normalised)
    )
    return df.loc[mask].copy()


def _write_empty_predictions(output_csv: Path, columns: Sequence[str]) -> None:
    cols = list(columns)
    if "prediction" not in cols:
        cols.append("prediction")
    pd.DataFrame(columns=cols).to_csv(output_csv, index=False)


def main(cfg: Settings) -> None:
    settings = cfg.reaction_prediction
    if not settings.data_csv:
        raise SystemExit("reaction_prediction.data_csv is not configured")
    if not settings.model_path:
        raise SystemExit("reaction_prediction.model_path is not configured")
    if not settings.output_csv:
        raise SystemExit("reaction_prediction.output_csv is not configured")

    data_csv = Path(settings.data_csv).expanduser().resolve()
    model_path = Path(settings.model_path).expanduser().resolve()
    output_csv = Path(settings.output_csv).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not data_csv.exists():
        raise FileNotFoundError(f"Data CSV not found: {data_csv}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    df = pd.read_csv(data_csv)
    df = _filter_trial_types(df, allowed=("testing",))
    if df.empty:
        print(
            "[REACTION] No testing trials found in data_csv; "
            "writing empty predictions spreadsheet."
        )
        _write_empty_predictions(output_csv, [])
        return

    span_threshold = float(getattr(cfg, "non_reactive_span_px", NON_REACTIVE_SPAN_PX))
    filtered_df, flagged_pairs = _drop_flagged_flies(df, threshold=span_threshold)
    if filtered_df.empty:
        print(
            "[REACTION] All candidate flies were flagged non-reactive; "
            "skipping prediction model and writing empty spreadsheet."
        )
        _write_empty_predictions(output_csv, df.columns)
        return

    data_csv_for_cli = data_csv
    temp_path: Path | None = None

    if len(filtered_df) != len(df):
        flagged_labels = ", ".join(
            f"{dataset}::{fly}::{fly_number}" for dataset, fly, fly_number in sorted(flagged_pairs)
        )
        print(
            f"[REACTION] Skipping non-reactive flies before prediction: {flagged_labels}"
        )
        tmp = tempfile.NamedTemporaryFile(
            "w", suffix="_non_reactive_filtered.csv", delete=False
        )
        try:
            filtered_df.to_csv(tmp.name, index=False)
        finally:
            tmp.close()
        data_csv_for_cli = Path(tmp.name)
        temp_path = data_csv_for_cli

    cmd = [
        "flybehavior-response",
        "predict",
        "--data-csv",
        str(data_csv_for_cli),
        "--model-path",
        str(model_path),
        "--output-csv",
        str(output_csv),
    ]

    repo_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    extra_paths = [str(repo_root)]
    if pythonpath:
        extra_paths.append(pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(extra_paths)

    try:
        print("[REACTION] Running flybehavior-response →", " ".join(cmd))
        subprocess.run(cmd, check=True, env=env)
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except OSError:
                pass

    print(f"[REACTION] Wrote predictions to {output_csv}")
