"""Pre vs post model scores on three windows: the odor, the time after it, both.

Built for IAA-Sensitivity5s-1-1, whose odor is on for only 5 s: a PER that
starts late can fall after odor-off and be missed by a score that looks only at
the odor. Every trial is rescored with the production model three times
(``window_rescore``):

    odor             the odor itself                 (5 s here)
    after            the AFTER_S seconds after odor-off  (5 s)
    odor_plus_after  both                            (10 s)

All three go through the same feature path, so the windows differ ONLY in the
window. (The ``odor`` window is the production score: with the deployed delay
model it reproduced model_predictions.csv on 100 % of 11,942 trials, 2026-11-10.)
A model with an odor-arrival delay (``fbr_odor_delay_s``, 1 s for the deployed
model) shifts every window by it, as production shifts its odor window, and
the figures say so under their title.

Pre vs post is paired within fly per odor, as in pretest_vs_test_comparison:
Wilcoxon signed-rank on the score, exact McNemar on responded (score >= 2);
n counts flies; p uncorrected. Pre-test orange, post-training blue.

Caveat: the model was trained on 30 s odor windows; a 5 s or 10 s window is an
input it never saw. Read differences between windows as "where in time the
model finds the response", not as calibrated scores.

Outputs, per cohort, under ``--out-dir/<cohort>/``:
    <cohort>_score_windows.png          2 x 3 grid (score / % responding x window)
    <cohort>_score_windows_cs_plus.png  the trained odor, windows side by side
    <cohort>_score_windows_summary.csv  per window x odor: means, %, p
    <cohort>_score_windows_trials.csv   per fly x odor: pre/post score per window
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts.analysis import envelope_visuals as ev  # noqa: E402
from scripts.analysis import phase_terms  # noqa: E402
from scripts.analysis import pretest_vs_test_comparison as pvt  # noqa: E402
from scripts.analysis import significance_brackets  # noqa: E402
from scripts.analysis import window_rescore as wr  # noqa: E402
from scripts.analysis.odor_conc_labels import odor_conc_label, with_conc  # noqa: E402
from scripts.analysis.pretest_vs_test_traces import (  # noqa: E402
    POST_COLOR,
    POST_LABEL,
    PRE_COLOR,
    PRE_LABEL,
    odor_name,
    odor_parts,
    order_odors,
)

WINDOWS = ("odor", "after", "odor_plus_after")
KEY = ["dataset_canon", "fly", "fly_number", "odor"]


def window_titles(odor_s: float, after_s: float) -> dict[str, str]:
    o, a = f"{odor_s:g}", f"{after_s:g}"
    return {
        "odor": f"Odor ({o} s)",
        "after": f"{a} s after odor-off",
        "odor_plus_after": f"Odor + {a} s after ({odor_s + after_s:g} s)",
    }


def delay_note(delay_s: float | None) -> str | None:
    """The title line naming a model's odor-arrival shift; None without one."""
    if not delay_s:
        return None
    return (f"Windows shifted +{float(delay_s):g} s from the valve times "
            "(the model's odor-arrival delay)")


def _with_delay(title: str, delay_s: float | None) -> str:
    note = delay_note(delay_s)
    return f"{title}\n{note}" if note else title


# ---------------------------------------------------------------------------
# Rescoring and pairing
# ---------------------------------------------------------------------------


def _pair(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per (fly, odor) with the pre and post score, light trial dropped."""
    df = scored.copy()
    df["dataset_canon"] = df["dataset"].map(ev._canon_dataset)
    df = ev._normalise_fly_columns(df)
    df["odor"] = [odor_name(d, t) for d, t in zip(df["dataset_canon"], df["trial_label"])]
    df["odor_base"] = [
        (odor_parts(d, t) or ("", ""))[0] for d, t in zip(df["dataset_canon"], df["trial_label"])
    ]
    df = df.loc[df["odor"].notna()]
    phase = df["trial_type"].astype(str).str.strip().str.lower()
    cols = KEY + ["odor_base", "score"]
    pre = df.loc[phase == "pretest", cols].drop_duplicates(subset=KEY)
    post = df.loc[phase == "testing", cols].drop_duplicates(subset=KEY)
    paired = pre.rename(columns={"score": "score_pre"}).merge(
        post.drop(columns="odor_base").rename(columns={"score": "score_post"}), on=KEY
    )
    return paired.reset_index(drop=True)


def paired_by_window(
    template: pd.DataFrame,
    *,
    wide_paths: Sequence[Path | str],
    model_path: Path | str,
    rule,
    after_s: float,
    binary_threshold: int,
) -> dict[str, pd.DataFrame]:
    """The paired pre/post table for each window, in ``WINDOWS`` order."""
    out = {}
    for name in WINDOWS:
        scored = wr.rescore_predictions(
            template, wide_paths, model_path, wr.window_spec(name, after_s),
            binary_threshold=binary_threshold, rule=rule,
        )
        out[name] = _pair(scored)
    return out


def summarise(paired: Mapping[str, pd.DataFrame], *, binary_threshold: int) -> pd.DataFrame:
    """Per window x odor: n flies, means / SEM, % responding, paired p-values."""
    frames = []
    for name, table in paired.items():
        s = pvt.per_odor_summary(table, binary_threshold=binary_threshold)
        order = order_odors(s["odor"])
        s = s.set_index("odor").loc[order].reset_index()
        base = table.drop_duplicates("odor").set_index("odor")["odor_base"]
        s.insert(0, "window", name)
        s["odor_base"] = s["odor"].map(base)
        frames.append(s)
    return pd.concat(frames, ignore_index=True)


def trials_table(paired: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Per fly x odor, the pre and post score under every window."""
    merged = None
    for name, table in paired.items():
        t = table[KEY + ["score_pre", "score_post"]].rename(columns={
            "score_pre": f"score_pre_{name}", "score_post": f"score_post_{name}",
        })
        merged = t if merged is None else merged.merge(t, on=KEY, how="outer")
    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _bars(ax, x, pre, post, err_pre=None, err_post=None, width=0.38, kind="score",
          cohort=None):
    gap = 0.012
    # A sensitivity cohort's odor is never paired with anything, so its panels
    # are pre-/post-exposure, not pre-test/post-training.
    pre_label = phase_terms.pre_label(cohort, PRE_LABEL)
    post_label = phase_terms.post_label(cohort, POST_LABEL)
    for offset, values, err, colour, label in (
        (-(width / 2 + gap), pre, err_pre, PRE_COLOR, pre_label),
        (+(width / 2 + gap), post, err_post, POST_COLOR, post_label),
    ):
        ax.bar(x + offset, values, width, yerr=err, color=colour, linewidth=0,
               label=label, error_kw=dict(ecolor=pvt.INK_MUTED, elinewidth=0.9,
                                          capsize=2.5, capthick=0.9))
        # A zero bar is otherwise indistinguishable from "no data" (same rule
        # as pretest_vs_test_comparison).
        for xi, value in zip(x + offset, values):
            if value == 0:
                ax.text(xi, pvt._zero_label_offset(kind), "0", ha="center",
                        va="bottom", fontsize=7.5, color=pvt.INK_MUTED)


def _n_flies(summary: pd.DataFrame) -> str:
    counts = sorted(set(int(v) for v in summary["n_flies"]))
    return f"n={counts[0]} flies" if len(counts) == 1 else f"n={counts[0]}–{counts[-1]} flies"


def _odor_labels(summary: pd.DataFrame, cohort: str) -> list[str]:
    """Drawn odor names, one per row: ``"Isoamyl Acetate 1% (v/v)"`` for a sensitivity cohort.

    ``odor`` ("Isoamyl Acetate 1") stays the key; its concentration is what it
    adds to ``odor_base``. Every other cohort's names are drawn as they are.
    """
    odors = [str(o) for o in summary["odor"]]
    if not phase_terms.is_sensitivity(cohort):
        return odors
    labels = []
    for odor, base in zip(odors, summary["odor_base"]):
        base = base if isinstance(base, str) else ""
        conc = odor[len(base):].strip() if base and odor.startswith(base) else ""
        labels.append(with_conc(base, conc) if conc else odor_conc_label(odor))
    return labels


def build_grid(summary: pd.DataFrame, *, cohort: str, titles: Mapping[str, str],
               delay_s: float | None = None):
    """Score (top) and % responding (bottom), one column per window. Unsaved.

    ``delay_s``: the model's odor-arrival delay, named under the title.
    """
    windows = [w for w in WINDOWS if w in set(summary["window"])]
    n_odors = summary.groupby("window").size().max()
    with plt.rc_context(pvt._RC_CONTEXT):
        fig, axes = plt.subplots(
            2, len(windows), figsize=(1.0 * n_odors * len(windows) + 2.5, 8.0),
            squeeze=False, layout="constrained",
        )
        for col, name in enumerate(windows):
            s = summary.loc[summary["window"] == name]
            x = np.arange(len(s))
            for row, (kind, pre, post, err, p_key) in enumerate((
                ("score", "mean_pre", "mean_post", ("sem_pre", "sem_post"), "p_score"),
                ("rate", "pct_pre", "pct_post", None, "p_rate"),
            )):
                ax = axes[row, col]
                _bars(ax, x, s[pre].to_numpy(float), s[post].to_numpy(float),
                      s[err[0]] if err else None, s[err[1]] if err else None, kind=kind,
                      cohort=cohort)
                (pvt._apply_score_axis if kind == "score" else pvt._apply_rate_axis)(ax)
                if col:
                    ax.set_ylabel("")
                ax.set_xticks(x)
                ax.set_xticklabels(_odor_labels(s, cohort), rotation=35, ha="right")
                ax.set_xlim(-0.6, len(s) - 0.4)
                significance_brackets.draw(ax, x, 2 * 0.38, list(s[p_key]), expand=False)
                if row == 0:
                    ax.set_title(titles[name], loc="left", fontweight="bold")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper right", ncol=2)
        fig.suptitle(
            _with_delay(
                f"{cohort} — model score by window, "
                f"{phase_terms.pre_label(cohort, 'pre-test', case='lower')} vs "
                f"{phase_terms.post_label(cohort, 'post-training', case='lower')}\n"
                f"{_n_flies(summary)}, paired within fly · Wilcoxon (score), "
                "exact McNemar (responded = score ≥ 2), uncorrected",
                delay_s,
            ),
            x=0.01, ha="left",
        )
    return fig


def build_cs_plus(summary: pd.DataFrame, *, cohort: str, titles: Mapping[str, str],
                  delay_s: float | None = None):
    """The trained odor only: windows side by side. None if it was not tested.

    ``delay_s``: the model's odor-arrival delay, named under the title.
    """
    trained = str(ev._trained_label(cohort)).strip().lower()
    cs = summary.loc[summary["odor_base"].astype(str).str.strip().str.lower() == trained]
    if cs.empty:
        return None
    cs = cs.set_index("window").loc[[w for w in WINDOWS if w in set(cs["window"])]]
    x = np.arange(len(cs))
    labels = [titles[w] for w in cs.index]
    with plt.rc_context(pvt._RC_CONTEXT):
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4), layout="constrained")
        for ax, (kind, pre, post, err, p_key) in zip(axes, (
            ("score", "mean_pre", "mean_post", ("sem_pre", "sem_post"), "p_score"),
            ("rate", "pct_pre", "pct_post", None, "p_rate"),
        )):
            _bars(ax, x, cs[pre].to_numpy(float), cs[post].to_numpy(float),
                  cs[err[0]] if err else None, cs[err[1]] if err else None, kind=kind,
                  cohort=cohort)
            (pvt._apply_score_axis if kind == "score" else pvt._apply_rate_axis)(ax)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_xlim(-0.6, len(cs) - 0.4)
            significance_brackets.draw(ax, x, 2 * 0.38, list(cs[p_key]), expand=False)
        axes[0].legend(loc="upper left")
        fig.suptitle(
            _with_delay(
                f"{cohort} — {_odor_labels(cs.iloc[:1], cohort)[0]} "
                f"({phase_terms.odor_tag(cohort)}), score by window\n"
                f"n={int(cs['n_flies'].min())} flies, paired within fly, uncorrected",
                delay_s,
            ),
            x=0.01, ha="left",
        )
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _odor_seconds(wide_paths: Sequence[Path | str], cohort: str) -> float:
    """The cohort's typical odor duration, for the panel titles."""
    import pyarrow.parquet as pq

    spans = []
    for path in wide_paths:
        t = pq.read_table(path, columns=["dataset", "trial_odor_on_s", "trial_odor_off_s"],
                          filters=[("dataset", "==", cohort)]).to_pandas()
        spans.append(t["trial_odor_off_s"] - t["trial_odor_on_s"])
    span = pd.concat(spans).dropna() if spans else pd.Series(dtype=float)
    return float(round(span.median())) if len(span) else 30.0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--predictions-csv", required=True, type=Path)
    p.add_argument("--wide-csv", required=True, nargs="+", type=Path)
    p.add_argument("--model-path", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--cohorts", required=True, nargs="+")
    p.add_argument("--after-s", type=float, default=5.0)
    p.add_argument("--binary-threshold", type=int, default=2)
    p.add_argument("--out-dir", required=True, type=Path)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rule = wr.threshold_rule_from_config(args.config)
    delay = wr.model_odor_delay_s(args.model_path)
    predictions = pd.read_csv(args.predictions_csv)
    for cohort in args.cohorts:
        template = predictions.loc[predictions["dataset"].astype(str) == cohort]
        if template.empty:
            print(f"[score-windows] {cohort}: no scored trials — skipped")
            continue
        paired = paired_by_window(
            template.reset_index(drop=True), wide_paths=args.wide_csv,
            model_path=args.model_path, rule=rule, after_s=args.after_s,
            binary_threshold=args.binary_threshold,
        )
        if all(t.empty for t in paired.values()):
            print(f"[score-windows] {cohort}: no paired pre/post trials — skipped")
            continue
        summary = summarise(paired, binary_threshold=args.binary_threshold)
        titles = window_titles(_odor_seconds(args.wide_csv, cohort), args.after_s)
        out = args.out_dir / ev._safe_dirname(cohort)
        out.mkdir(parents=True, exist_ok=True)
        stem = ev._safe_dirname(cohort)
        fig = build_grid(summary, cohort=cohort, titles=titles, delay_s=delay)
        fig.savefig(out / f"{stem}_score_windows.png")
        plt.close(fig)
        fig = build_cs_plus(summary, cohort=cohort, titles=titles, delay_s=delay)
        if fig is not None:
            fig.savefig(out / f"{stem}_score_windows_cs_plus.png")
            plt.close(fig)
        summary.assign(window_title=summary["window"].map(titles)).to_csv(
            out / f"{stem}_score_windows_summary.csv", index=False)
        trials_table(paired).to_csv(out / f"{stem}_score_windows_trials.csv", index=False)
        print(f"[score-windows] {cohort}: wrote figures + CSVs → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
