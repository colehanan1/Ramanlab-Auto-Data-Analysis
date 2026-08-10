"""Score a group-mean trace with the ordinal PER model.

The model normally scores one trial of one fly. To put a number on a *cohort
mean* trace we synthesise a "mean fly" row and hand it to the same
``flybehavior-response predict-ordinal`` CLI the pipeline uses, so the scoring
path is identical to the one that produced ``model_predictions.csv``.

How the synthetic row is built, and why
---------------------------------------
The model's 24 features split in two:

* **13 signal features** — recomputed by the model itself from the ``dir_val_*``
  trace. These come from the mean trace directly, which is the part of this
  question that actually has an answer.
* **11 engineered features** — precomputed upstream. Four of them
  (``global_max``, ``trimmed_global_min`` and friends) are *per-fly* properties
  measured on the raw class-2 distance signal, not on the trace, and have no
  definition for an average of several flies. The rest depend on a fly-level
  baseline median that likewise is not stored per trace.

So the engineered block is **averaged across the contributing flies** rather
than recomputed. Averaging cannot silently diverge from the pipeline's
definitions the way a reimplementation could, and it is the natural "mean fly"
analogue. The trace-derived half is genuinely the mean trace.

Treat the result as descriptive, not inferential: the model never saw averaged
traces in training, and a cohort mean is smoother than any real trial, which
generally pushes scores toward the middle of the range.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Mapping, Sequence

import pandas as pd

LOGGER = logging.getLogger("mean_trace_score")

DEFAULT_BINARY_THRESHOLD = 2

# Metadata carried through to the CLI output so rows can be matched back.
_ID_COLUMNS = ("dataset", "fly", "fly_number", "trial_label")


def build_mean_row(rows: pd.DataFrame) -> pd.Series:
    """Collapse the contributing flies' wide rows into one "mean fly" row.

    Every numeric column — the ``dir_val_*`` trace, the engineered features and
    the per-trial timing — is averaged with NaNs skipped. Non-numeric columns
    take the first row's value, so ``dataset`` / ``trial_type`` survive intact.
    """
    if rows.empty:
        raise ValueError("Cannot build a mean row from zero rows")
    numeric = rows.select_dtypes(include="number")
    mean = numeric.mean(axis=0, skipna=True)
    out = rows.iloc[0].copy()
    out[mean.index] = mean
    return out


def _default_runner(cmd: Sequence[str]) -> None:
    LOGGER.debug("running %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)


def score_rows(
    rows: pd.DataFrame,
    *,
    model_path: Path,
    binary_threshold: int = DEFAULT_BINARY_THRESHOLD,
    runner: Callable[[Sequence[str]], None] | None = None,
    cli: str = "flybehavior-response",
) -> pd.DataFrame:
    """Score wide-format rows via the pipeline's ordinal-model CLI.

    Returns the CLI's output frame (``score`` / ``prediction`` plus whichever
    id columns were present), in input order.
    """
    if rows.empty:
        return pd.DataFrame(columns=["score", "prediction"])
    run = runner or _default_runner
    with tempfile.TemporaryDirectory() as tmp:
        data_csv = Path(tmp) / "mean_rows.csv"
        out_csv = Path(tmp) / "mean_scores.csv"
        rows.to_csv(data_csv, index=False)
        run([
            cli, "predict-ordinal",
            "--data-csv", str(data_csv),
            "--model-path", str(model_path),
            "--output-csv", str(out_csv),
            "--binary-threshold", str(binary_threshold),
        ])
        if not out_csv.exists():
            raise RuntimeError(f"{cli} produced no output for {len(rows)} rows")
        return pd.read_csv(out_csv)


def score_group_means(
    groups: Mapping[str, pd.DataFrame],
    *,
    model_path: Path,
    binary_threshold: int = DEFAULT_BINARY_THRESHOLD,
    runner: Callable[[Sequence[str]], None] | None = None,
    cli: str = "flybehavior-response",
) -> dict[str, dict[str, float]]:
    """Score one mean trace per group.

    ``groups`` maps a caller-chosen key (e.g. ``"Trained|Hexanol (0.1%)"``) to
    the wide rows contributing to that group's mean. Groups with no rows are
    skipped rather than scored, since an empty mean is not a trace.
    """
    keys: list[str] = []
    mean_rows: list[pd.Series] = []
    for key, rows in groups.items():
        if rows is None or rows.empty:
            LOGGER.warning("No rows for %s — not scoring", key)
            continue
        row = build_mean_row(rows)
        # Overwrite the id columns so each synthetic row is traceable in the
        # CLI output; `fly` doubles as the group key.
        if "fly" in row.index:
            row["fly"] = key
        if "fly_number" in row.index:
            row["fly_number"] = 0
        keys.append(key)
        mean_rows.append(row)

    if not mean_rows:
        return {}

    frame = pd.DataFrame(mean_rows).reset_index(drop=True)
    scored = score_rows(
        frame, model_path=model_path, binary_threshold=binary_threshold,
        runner=runner, cli=cli,
    )
    if len(scored) != len(keys):
        raise RuntimeError(
            f"Scored {len(scored)} rows for {len(keys)} groups — output misaligned"
        )
    out: dict[str, dict[str, float]] = {}
    for key, (_, row) in zip(keys, scored.iterrows()):
        out[key] = {
            "score": int(row["score"]),
            "prediction": int(row["prediction"]),
            "n_flies": int(len(groups[key])),
        }
    return out


def model_settings_from_config(config_path: str | Path) -> tuple[Path, int]:
    """``(model_path, binary_threshold)`` from a pipeline config."""
    repo_root = Path(__file__).resolve().parents[2]
    for p in (str(repo_root), str(repo_root / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from fbpipe.config import load_settings

    settings = load_settings(str(config_path))
    reaction = getattr(settings, "reaction_prediction", None)
    model_path = getattr(reaction, "model_path", "") if reaction else ""
    if not model_path:
        raise ValueError(f"reaction_prediction.model_path not set in {config_path}")
    threshold = getattr(reaction, "binary_threshold", DEFAULT_BINARY_THRESHOLD)
    return Path(model_path), int(threshold)


def annotate_mean_scores(ax, trained: dict, control: dict) -> None:
    """Print the model's score for each cohort's mean trace on the axes.

    Placed bottom-left, away from the legend (upper right) and from the odor
    window. Labelled "mean trace" because it is not the mean of the per-fly
    scores — see this module's docstring for what the model is actually fed.
    """
    lines = [
        "Model score of mean trace",
        f"  Trained (n={trained['n_flies']}):  {trained['score']}"
        f"  [{'PER' if trained['prediction'] else 'no PER'}]",
        f"  Control (n={control['n_flies']}):  {control['score']}"
        f"  [{'PER' if control['prediction'] else 'no PER'}]",
    ]
    ax.text(
        0.015, 0.03, "\n".join(lines), transform=ax.transAxes,
        va="bottom", ha="left", fontsize=8.5, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="0.65", alpha=0.92),
        zorder=6,
    )
