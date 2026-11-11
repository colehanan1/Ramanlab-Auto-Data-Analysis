#!/usr/bin/env python3
"""Pre-test vs post-training PER across two starvation groups (e.g. 12 h vs 24 h).

Each group is one ``*-Sensitivity-*`` cohort, optionally narrowed to the flies
whose folder starts with a prefix (``24 h=IAA-Sensitivity-24-1:october_06`` is
only the october_06 batches of the 24 h cohort). Within a group the comparison
is paired per fly, exactly as in ``pretest_vs_test_comparison.py``; between the
groups it is unpaired, because they are different flies. The response rate is
the exception: pre vs post within a group is ALSO Fisher's exact (the 2x2 phase
x responded table), at the user's request, rather than the paired McNemar.

* score: Mann-Whitney U (two-sided) on the per-fly scores, pre and post
  separately, and on the per-fly change (post - pre);
* response rate (score >= ``binary_threshold``): Fisher's exact test.

p-values are uncorrected, matching the score_summary and pre-vs-post figures.
Each fly meets each odor once per phase, so n counts flies and a group mean is
the mean over its flies' trials -- there is no average of averages anywhere.

Trial labels recorded under ``~/odor_pins.txt`` (2026-10-01 on) carry the vial
concentration ("testing_4_3octanol-0.1"), which the shared figure code does not
parse yet. :func:`normalize_trial_label` strips it back to the legacy rig
spelling so both groups land in the same odor column. Only use it on cohorts
whose panel holds each odor at ONE concentration -- the hex-test panel carries
two hexanol vials, which this would merge. A sensitivity figure draws the
concentration back into the name ("Isoamyl Acetate 1% (v/v)", :func:`odor_text`);
the tables keep the normalised odor.

Two or three groups. With three (e.g. 12 h / 18 h / 24 h) every pair is tested
as above and ``stats_omnibus.csv`` adds Kruskal-Wallis (scores) and a
Fisher-Freeman-Halton exact test (response rate) across all three.

Colour: hue is the phase (pre-test blue / post-training orange, as in every
pre-vs-post figure); lightness is the starvation group, lighter = the first
(less starved) group. Both two-step ramps pass the dataviz validator
(``--ordinal``) and the four together pass the categorical checks; the light
steps sit under 3:1 on white, so every bar pair also carries a group label.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import fisher_exact, kruskal, mannwhitneyu  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.analysis import odor_bar_palette, phase_terms, significance_brackets  # noqa: E402
from scripts.analysis.odor_conc_labels import odor_conc_label, with_conc  # noqa: E402
from scripts.analysis.pretest_vs_test_comparison import (  # noqa: E402
    BINARY_THRESHOLD,
    INK,
    INK_MUTED,
    POST_LABEL,
    PRE_LABEL,
    _RC_CONTEXT,
    _apply_rate_axis,
    _apply_score_axis,
    _recessive_grid,
    _sem,
    load_paired_scores,
    per_odor_summary,
)

# ── Labels ────────────────────────────────────────────────────────────────

_LABEL_RE = re.compile(r"^(?P<prefix>(?:pretest|testing|training)_\d+_)(?P<odor>.+)$")
_CONC_RE = re.compile(r"^(?P<odor>.+?)-(?P<conc>\d+(?:\.\d+)?)$")
#: odor_pins spellings that the display table does not know, mapped to the
#: legacy rig spelling it does ("3-octonol" -> "3-Octanol").
_SPELLING = {"3octanol": "3-octonol", "3-octanol": "3-octonol", "3octonol": "3-octonol"}


def _split_label(label: object) -> tuple[str, str | None, str | None]:
    text = str(label).strip().lower()
    m = _LABEL_RE.match(text)
    if not m:
        return text, None, None
    odor, conc = m["odor"], None
    c = _CONC_RE.match(odor)
    if c:
        odor, conc = c["odor"], c["conc"]
    return m["prefix"], _SPELLING.get(odor, odor), conc


def normalize_trial_label(label: object) -> str:
    """``testing_4_3octanol-0.1`` -> ``testing_4_3-octonol`` (lower case)."""
    prefix, odor, _conc = _split_label(label)
    return prefix if odor is None else f"{prefix}{odor}"


def label_concentration(label: object) -> str | None:
    """The vial concentration a pins-file label carries, or None."""
    return _split_label(label)[2]


#: The pipeline's live predictions table. The trend and findings scripts read a
#: comparison folder, whose tables keep only the normalised labels, so by
#: default they look the recorded labels (the concentrations) up here.
PREDICTIONS_CSV = Path("/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/model_predictions.csv")
_FLY_KEY = ("dataset_canon", "fly", "fly_number")


def odor_concentrations(preds, flies: pd.DataFrame | None = None) -> dict[str, str]:
    """Display odor -> the vial concentration its trial labels record ("0.1").

    ``preds``: prediction rows with the labels as recorded, before
    ``subset_and_relabel`` strips the concentration -- a frame, or a CSV path
    (a missing or empty path gives {}). ``flies`` (dataset_canon / fly /
    fly_number, e.g. the paired table) narrows them to the flies a figure draws.
    An odor seen at two concentrations, or with and without one (a batch
    recorded before the pins file), gets no entry: it is drawn without a
    percent rather than with a guessed one.
    """
    from fbpipe.config import canon_fly_number
    from scripts.analysis.envelope_visuals import (
        _canon_dataset,
        _display_label_ci,
        _extract_odor_from_label,
        apply_dataset_odor_remap,
    )
    from scripts.analysis.score_summary import _normalise_trial_label

    if not isinstance(preds, pd.DataFrame):
        if not preds or not Path(preds).is_file():
            return {}
        cols = {"dataset", "fly", "fly_number", "trial_label", "trial_type"}
        preds = pd.read_csv(preds, usecols=lambda c: c in cols)
    if "trial_type" in preds:  # the drawn phases only
        preds = preds[preds["trial_type"].astype(str).str.strip().str.lower().isin(("pretest", "testing"))]
    datasets = preds["dataset"].astype(str).str.strip().map(_canon_dataset)
    if flies is not None:
        if not set(_FLY_KEY) <= set(flies.columns):
            return {}
        drawn = set(zip(flies["dataset_canon"].astype(str), flies["fly"].astype(str).str.strip(),
                        flies["fly_number"].map(canon_fly_number)))
        keep = [k in drawn for k in zip(datasets, preds["fly"].astype(str).str.strip(),
                                        preds["fly_number"].map(canon_fly_number))]
        preds, datasets = preds[keep], datasets[keep]
    if preds.empty:
        return {}
    labels = preds["trial_label"].astype(str)
    # Named exactly as _load_scores names the paired table's odors.
    odors = [apply_dataset_odor_remap(d, _display_label_ci(_extract_odor_from_label(
                 _normalise_trial_label(normalize_trial_label(lab)))))
             for d, lab in zip(datasets, labels)]
    out = {}
    for odor, concs in labels.map(label_concentration).groupby(odors):
        if concs.notna().all() and len({float(c) for c in concs}) == 1:
            out[odor] = concs.iloc[0]
    return out


def odor_text(odor: str, cohort: str | None, concentrations=None) -> str:
    """How ``odor`` is drawn: "<Odor> <c>% (v/v)" on a sensitivity-cohort figure.

    ``concentrations`` (``odor_concentrations``) supplies the percent; an odor
    without one is drawn by its display name alone. Every other cohort's figure
    keeps ``odor`` unchanged. Display only: grouping, sorting and trained-odor
    matching keep using ``odor``.
    """
    if not phase_terms.is_sensitivity(cohort):
        return odor
    conc = (concentrations or {}).get(odor)
    return with_conc(odor, conc) if conc else odor_conc_label(odor)


# ── Groups ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Group:
    label: str
    dataset: str
    fly_prefix: str = ""


def parse_group(spec: str) -> Group:
    """``"24 h=IAA-Sensitivity-24-1:october_06"`` -> Group."""
    if "=" not in spec:
        raise ValueError(f"group spec needs LABEL=DATASET[:FLY_PREFIX]: {spec!r}")
    label, rest = spec.split("=", 1)
    dataset, _, prefix = rest.partition(":")
    if not label.strip() or not dataset.strip():
        raise ValueError(f"group spec needs LABEL=DATASET[:FLY_PREFIX]: {spec!r}")
    return Group(label.strip(), dataset.strip(), prefix.strip())


def select_group_rows(df: pd.DataFrame, group: Group) -> pd.DataFrame:
    mask = df["dataset"].astype(str).str.strip() == group.dataset
    if group.fly_prefix:
        mask &= df["fly"].astype(str).str.startswith(_prefixes(group))
    return df.loc[mask]


def _prefixes(group: Group) -> tuple[str, ...]:
    """``fly_prefix`` may list several, comma-separated (october_06,october_07)."""
    return tuple(p.strip() for p in group.fly_prefix.split(",") if p.strip())


def subset_and_relabel(df: pd.DataFrame, groups: Sequence[Group]) -> pd.DataFrame:
    """Rows of every group, trial labels normalised. Dataset names are kept."""
    parts = [select_group_rows(df, g) for g in groups]
    out = pd.concat(parts) if parts else df.iloc[0:0]
    out = out.copy()
    out["trial_label"] = out["trial_label"].map(normalize_trial_label)
    return out


def drop_flagged(df: pd.DataFrame, flagged_flies_csv: str | Path) -> pd.DataFrame:
    """Drop every row of a fly the truth table marks FLY-State 0 or -1.

    Same rule and parser as the pipeline (``load_flagged_fly_exclusions``).
    Applied to the input tables themselves, because not every figure script
    reads the table: some trust model_predictions.csv to have dropped the fly
    already, which is only true if it was scored after the fly was flagged.
    """
    from fbpipe.config import canon_fly_number, load_flagged_fly_exclusions

    if not flagged_flies_csv or not Path(flagged_flies_csv).exists():
        return df
    excluded = load_flagged_fly_exclusions(flagged_flies_csv)
    if not excluded:
        return df
    keys = [
        (str(d).strip(), str(f).strip(), canon_fly_number(n))
        for d, f, n in zip(df["dataset"], df["fly"], df["fly_number"])
    ]
    return df.loc[[k not in excluded for k in keys]]


def assign_groups(paired: pd.DataFrame, groups: Sequence[Group]) -> pd.DataFrame:
    """Tag each paired row with its group label; rows in no group are dropped."""
    out = paired.copy()
    out["group"] = None
    for g in groups:
        mask = out["dataset_canon"].astype(str) == g.dataset
        if g.fly_prefix:
            mask &= out["fly"].astype(str).str.startswith(_prefixes(g))
        out.loc[mask, "group"] = g.label
    return out.dropna(subset=["group"]).reset_index(drop=True)


# ── Statistics ────────────────────────────────────────────────────────────


def group_summary(
    paired: pd.DataFrame, groups: Sequence[str], *, binary_threshold: int = BINARY_THRESHOLD
) -> pd.DataFrame:
    """Per (group, odor): n flies, pre/post means and rates, and p-values.

    Score: Wilcoxon signed-rank (paired). Response rate: Fisher's exact.
    """
    parts = []
    for g in groups:
        s = per_odor_summary(paired[paired["group"] == g], binary_threshold=binary_threshold,
                             rate_test="fisher")
        s.insert(0, "group", g)
        parts.append(s)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _mwu(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size == 0 or b.size == 0:
        return None
    try:
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError:
        return None


def _fisher(resp_a: np.ndarray, resp_b: np.ndarray) -> float | None:
    if resp_a.size == 0 or resp_b.size == 0:
        return None
    ra, rb = int(resp_a.sum()), int(resp_b.sum())
    return float(fisher_exact([[ra, resp_a.size - ra], [rb, resp_b.size - rb]]).pvalue)


def between_group_stats(
    paired: pd.DataFrame, group_a: str, group_b: str, *, binary_threshold: int = BINARY_THRESHOLD
) -> pd.DataFrame:
    """Unpaired group_a vs group_b per odor, for pre, post and the change."""
    a_all = paired[paired["group"] == group_a]
    b_all = paired[paired["group"] == group_b]
    rows = []
    for odor in sorted(set(a_all["odor"]) & set(b_all["odor"])):
        a = a_all[a_all["odor"] == odor]
        b = b_all[b_all["odor"] == odor]
        vals = {}
        for name, frame in (("a", a), ("b", b)):
            pre = frame["score_pre"].to_numpy(dtype=float)
            post = frame["score_post"].to_numpy(dtype=float)
            vals[name] = (pre, post, post - pre)
        (pa, qa, da), (pb, qb, db) = vals["a"], vals["b"]
        rows.append({
            "odor": odor,
            "group_a": group_a,
            "group_b": group_b,
            "n_a": pa.size,
            "n_b": pb.size,
            "mean_delta_a": float(np.mean(da)),
            "sem_delta_a": _sem(da),
            "mean_delta_b": float(np.mean(db)),
            "sem_delta_b": _sem(db),
            "p_pre_score": _mwu(pa, pb),
            "p_post_score": _mwu(qa, qb),
            "p_delta_score": _mwu(da, db),
            "p_pre_rate": _fisher(pa >= binary_threshold, pb >= binary_threshold),
            "p_post_rate": _fisher(qa >= binary_threshold, qb >= binary_threshold),
        })
    return pd.DataFrame(rows)


def pairwise_between_stats(
    paired: pd.DataFrame, groups: Sequence[str], *, binary_threshold: int = BINARY_THRESHOLD
) -> pd.DataFrame:
    """:func:`between_group_stats` for every pair of groups, in group order."""
    parts = [between_group_stats(paired, a, b, binary_threshold=binary_threshold)
             for a, b in combinations(groups, 2)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _kw(samples: list[np.ndarray]) -> float | None:
    if any(s.size == 0 for s in samples):
        return None
    try:
        return float(kruskal(*samples).pvalue)
    except ValueError:  # every value identical
        return None


def _ffh(responded: list[np.ndarray]) -> float | None:
    from scripts.analysis.randompanel_conc_comparison import fisher_freeman_halton_mc

    if any(r.size == 0 for r in responded):
        return None
    cats = np.concatenate([r.astype(int) for r in responded])
    labels = np.concatenate([np.full(r.size, i) for i, r in enumerate(responded)])
    return float(fisher_freeman_halton_mc(cats, labels, n_iter=20_000))


def omnibus_stats(
    paired: pd.DataFrame, groups: Sequence[str], *, binary_threshold: int = BINARY_THRESHOLD
) -> pd.DataFrame:
    """Per odor, all groups at once: Kruskal-Wallis (scores), Fisher-Freeman-Halton (rate)."""
    by = {g: paired[paired["group"] == g] for g in groups}
    shared = set.intersection(*(set(f["odor"]) for f in by.values())) if by else set()
    rows = []
    for odor in sorted(shared):
        pre = [by[g].loc[by[g]["odor"] == odor, "score_pre"].to_numpy(dtype=float) for g in groups]
        post = [by[g].loc[by[g]["odor"] == odor, "score_post"].to_numpy(dtype=float) for g in groups]
        rows.append({
            "odor": odor,
            **{f"n_{g}": a.size for g, a in zip(groups, pre)},
            "p_kw_pre_score": _kw(pre),
            "p_kw_post_score": _kw(post),
            "p_kw_delta_score": _kw([b - a for a, b in zip(pre, post)]),
            "p_ffh_pre_rate": _ffh([a >= binary_threshold for a in pre]),
            "p_ffh_post_rate": _ffh([b >= binary_threshold for b in post]),
        })
    return pd.DataFrame(rows)


def check_group_count(groups: Sequence[Group]) -> None:
    """At least two groups; no upper limit (past six, ramps are sampled -- see group_colors)."""
    if len(groups) < 2:
        raise SystemExit(f"at least two --group specs are required, got {len(groups)}")


# ── Colour ────────────────────────────────────────────────────────────────

#: (pre, post) per group position: lighter = first (less starved) group.
GROUP_COLORS = (("#6da7ec", "#f0915f"), ("#1c5cab", "#a8401a"))
#: Three-step ramps for three groups: each validated --ordinal, and the six
#: together pass the categorical checks (light steps under 3:1 -> group labels).
GROUP_COLORS_3 = (("#6da7ec", "#f0915f"), ("#2a78d6", "#d95926"), ("#184f95", "#9a3a16"))
#: Change (post - pre) belongs to neither phase, so it wears neutral ink steps.
DELTA_COLORS = ("#9a9994", "#3d3c39")
DELTA_COLORS_3 = ("#b3b2ad", "#77766f", "#3d3c39")
#: Four/five-step ramps, each validated --ordinal (light end >= 2:1 on white).
_BLUE5 = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b")
_ORANGE5 = ("#f0915f", "#e5703f", "#c9501f", "#9a3a16", "#62230b")
_GRAY5 = ("#a8a7a1", "#86857f", "#66655f", "#46453f", "#262522")
_PICK = {4: (0, 1, 3, 4), 5: (0, 1, 2, 3, 4)}
#: Six-step ramps for six groups (e.g. 1/6/12/18/24/30 h), each validated
#: --ordinal (light end >= 2:1 on white). Separate from the 5-step ramps so
#: adding a sixth group never repaints the existing 4/5-group figures.
_BLUE6 = ("#86b6ef", "#5f9fea", "#3b86df", "#2a6fc4", "#1c5299", "#0d366b")
_ORANGE6 = ("#f29a6a", "#e5703f", "#d0582a", "#b0441b", "#863212", "#62230b")
_GRAY6 = ("#a8a7a1", "#8c8b85", "#706f69", "#55544f", "#3b3a36", "#262522")


def _sample_ramp(anchors: Sequence[str], n: int) -> tuple[str, ...]:
    """n steps evenly along a validated ramp, keeping its light and dark ends.

    Past six groups the steps sit closer than the dataviz ordinal check asks
    for, so shade is a secondary cue only: every bar pair is also labelled with
    its group underneath (``_group_ticks``).
    """
    from matplotlib.colors import LinearSegmentedColormap, to_hex

    cmap = LinearSegmentedColormap.from_list("ramp", list(anchors))
    return tuple(to_hex(cmap(i / (n - 1))) for i in range(n))


def group_colors(n: int):
    """(pre, post) colour per group position, light (least starved) to dark."""
    if n > 6:
        return tuple(zip(_sample_ramp(_BLUE6, n), _sample_ramp(_ORANGE6, n)))
    if n == 6:
        return tuple(zip(_BLUE6, _ORANGE6))
    if n in _PICK:
        return tuple((_BLUE5[i], _ORANGE5[i]) for i in _PICK[n])
    return {2: GROUP_COLORS, 3: GROUP_COLORS_3}[n]


def delta_colors(n: int):
    if n > 6:
        return _sample_ramp(_GRAY6, n)
    if n == 6:
        return _GRAY6
    if n in _PICK:
        return tuple(_GRAY5[i] for i in _PICK[n])
    return {2: DELTA_COLORS, 3: DELTA_COLORS_3}[n]


def _odor_order(summary: pd.DataFrame) -> list[str]:
    return sorted(summary["odor"].unique(), key=str.casefold)


def _tint_trained(ax, odors: Sequence[str], trained: str | None) -> None:
    if not trained:
        return
    for label, odor in zip(ax.get_xticklabels(), odors):
        if str(odor).casefold().startswith(trained.casefold()):
            label.set_color(odor_bar_palette.trained_tick_color(trained))
            label.set_fontweight("bold")


def _group_ticks(ax, centres: Sequence[float], text: str) -> None:
    """Small group names under each bar pair: identity never rests on colour."""
    for xc in centres:
        ax.text(xc, -0.012, text, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=6.5, color=INK_MUTED)


def _n_text(summary: pd.DataFrame, groups: Sequence[str]) -> str:
    parts = []
    for g in groups:
        counts = sorted(set(int(v) for v in summary.loc[summary["group"] == g, "n_flies"]))
        if not counts:
            continue
        n = f"{counts[0]}" if len(counts) == 1 else f"{counts[0]}–{counts[-1]}"
        parts.append(f"{g}: n={n} flies")
    return " · ".join(parts)


def _comparison(groups: Sequence[str], comparison: str | None) -> str:
    """Title phrase naming the two groups; starvation wording unless overridden."""
    return comparison or " vs ".join(groups) + " starved"


def _phase_cohort(cohorts) -> str | None:
    """The cohort name ``phase_terms`` words a figure by, or None.

    A sensitivity cohort only when every cohort drawn is one: a figure that
    mixes in a conditioning cohort keeps pre-test / post-training / CS+.
    """
    names = sorted({str(c) for c in cohorts if pd.notna(c)})
    return names[0] if phase_terms.all_sensitivity(names) else None


def _phase_names(cohort: str | None) -> tuple[str, str]:
    """(pre, post) panel names: Pre-/Post-exposure for a sensitivity cohort."""
    return phase_terms.pre_label(cohort, PRE_LABEL), phase_terms.post_label(cohort, POST_LABEL)


def _row(summary, group, odor):
    return summary[(summary["group"] == group) & (summary["odor"] == odor)]


def _value(summary, group, odor, key):
    row = _row(summary, group, odor)
    return (float(row[key].iloc[0]), row) if len(row) else (np.nan, row)


def _pair_row(between: pd.DataFrame, odor: str, a: str, b: str) -> pd.DataFrame:
    if between.empty:
        return between
    return between[(between["odor"] == odor) & (between["group_a"] == a)
                   & (between["group_b"] == b)]


def _pair_brackets(ax, odors, positions, groups, between, key) -> None:
    """Bracket every significant group pair per odor, narrowest pair first.

    ``positions[g][k]`` is group g's bar centre for odor k. Each call to the
    shared helper measures what is already drawn (bars, whiskers, earlier
    brackets and stars), so wider pairs stack above narrower ones.
    """
    order = sorted(combinations(range(len(groups)), 2), key=lambda ij: (ij[1] - ij[0], ij[0]))
    for k, odor in enumerate(odors):
        for i, j in order:
            p = _p(_pair_row(between, odor, groups[i], groups[j]), key)
            if not significance_brackets.stars(p):
                continue
            xi, xj = positions[groups[i]][k], positions[groups[j]][k]
            significance_brackets.draw(ax, [(xi + xj) / 2], xj - xi, [p], expand=False)


def _p(frame: pd.DataFrame, key: str) -> float | None:
    """First p-value in ``frame[key]``; None for an absent row or an untestable pair."""
    if not len(frame):
        return None
    p = frame[key].iloc[0]
    return None if p is None or pd.isna(p) else float(p)


def build_pre_post_bars(summary, *, groups, kind, trained_odor, title, comparison=None,
                        cohorts=(), concentrations=None):
    """Per odor: [g1 pre, g1 post] [g2 pre, g2 post]; paired brackets per group.

    ``cohorts`` (the datasets drawn) picks the phase wording, see ``_phase_cohort``;
    ``concentrations`` (``odor_concentrations``) the percent in each odor name.
    """
    is_score = kind == "score"
    pre_key, post_key = ("mean_pre", "mean_post") if is_score else ("pct_pre", "pct_post")
    p_key = "p_score" if is_score else "p_rate"
    cohort = _phase_cohort(cohorts)
    pre_name, post_name = _phase_names(cohort)
    odors = _odor_order(summary)
    x = np.arange(len(odors), dtype=float)
    n = len(groups)
    colors = group_colors(n)
    slot = 0.84 / n                    # x-width of one group's pre/post pair
    bar_w = slot / 2 - 0.02

    fig, ax = plt.subplots(figsize=(0.85 * n * max(len(odors), 1) + 2.2, 4.6))
    handles = []
    for gi, g in enumerate(groups):
        centre = x + (gi - (n - 1) / 2) * slot
        for phase, key, err, sign in (("pre", pre_key, "sem_pre", -1), ("post", post_key, "sem_post", 1)):
            color = colors[gi][0 if phase == "pre" else 1]
            vals, errs = [], []
            for odor in odors:
                v, row = _value(summary, g, odor, key)
                vals.append(v)
                errs.append(float(row[err].iloc[0]) if is_score and len(row) else 0.0)
            ax.bar(centre + sign * (bar_w / 2 + 0.005), vals, bar_w,
                   yerr=errs if is_score else None, color=color, linewidth=0,
                   error_kw=dict(ecolor=INK_MUTED, elinewidth=0.8, capsize=2, capthick=0.8))
            handles.append(Patch(color=color, label=f"{g} — {pre_name if phase == 'pre' else post_name}"))
        pvals = [_p(_row(summary, g, odor), p_key) for odor in odors]
        _group_ticks(ax, centre, g)
        (_apply_score_axis if is_score else _apply_rate_axis)(ax)
        significance_brackets.draw(ax, centre, 2 * bar_w, pvals, expand=False)

    ax.set_xticks(x)
    ax.set_xticklabels([odor_text(o, cohort, concentrations) for o in odors], rotation=28, ha="right")
    ax.tick_params(axis="x", pad=13, length=0)
    ax.set_xlim(-0.6, len(odors) - 0.4)
    _tint_trained(ax, odors, trained_odor)
    metric = "Mean PER score" if is_score else "PER response rate"
    test = ("paired within fly (Wilcoxon signed-rank, uncorrected)" if is_score
            else "pre vs post: Fisher's exact (uncorrected)")
    phases = (f"{phase_terms.pre_label(cohort, 'pre-test', case='lower')} vs "
              f"{phase_terms.post_label(cohort, 'post-training', case='lower')}")
    ax.set_title(f"{title} — {metric}, {phases}, {_comparison(groups, comparison)}\n"
                 f"{_n_text(summary, groups)} · {test}",
                 loc="left", color=INK, pad=8)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), ncol=1,
              handlelength=1.1, handleheight=0.9)
    return fig


def build_by_phase_bars(summary, between, *, groups, kind, trained_odor, title, comparison=None,
                        cohorts=(), concentrations=None):
    """Two panels (pre-test | post-training); group vs group within each."""
    is_score = kind == "score"
    cohort = _phase_cohort(cohorts)
    pre_name, post_name = _phase_names(cohort)
    odors = _odor_order(summary)
    x = np.arange(len(odors), dtype=float)
    n = len(groups)
    colors = group_colors(n)
    bar_w = 0.72 / n
    width = 0.525 * n * max(len(odors), 1) + 1.2
    if n == 2:  # side by side
        fig, axes = plt.subplots(1, 2, figsize=(2 * width, 4.4))
    else:       # stacked: three groups side by side twice would be too wide to read
        fig, axes = plt.subplots(2, 1, figsize=(width + 1.2, 8.6))
    for ai, (ax, phase) in enumerate(zip(axes, ("pre", "post"))):
        key = (f"mean_{phase}" if is_score else f"pct_{phase}")
        handles = []
        positions = {}
        for gi, g in enumerate(groups):
            color = colors[gi][ai]
            vals, errs = [], []
            for odor in odors:
                v, row = _value(summary, g, odor, key)
                vals.append(v)
                errs.append(float(row[f"sem_{phase}"].iloc[0]) if is_score and len(row) else 0.0)
            pos = x + (gi - (n - 1) / 2) * (bar_w + 0.012)
            positions[g] = pos
            ax.bar(pos, vals, bar_w, yerr=errs if is_score else None, color=color, linewidth=0,
                   error_kw=dict(ecolor=INK_MUTED, elinewidth=0.8, capsize=2, capthick=0.8))
            _group_ticks(ax, pos, g)
            handles.append(Patch(color=color, label=g))
        (_apply_score_axis if is_score else _apply_rate_axis)(ax)
        p_col = f"p_{phase}_{'score' if is_score else 'rate'}"
        _pair_brackets(ax, odors, positions, groups, between, p_col)
        ax.set_xticks(x)
        ax.set_xticklabels([odor_text(o, cohort, concentrations) for o in odors],
                           rotation=28, ha="right")
        ax.tick_params(axis="x", pad=13, length=0)
        ax.set_xlim(-0.6, len(odors) - 0.4)
        _tint_trained(ax, odors, trained_odor)
        ax.set_title(pre_name if phase == "pre" else post_name, loc="left", color=INK)
        ax.legend(handles=handles, loc="upper right", handlelength=1.1, handleheight=0.9)
        if ai == 1 and n == 2:
            ax.set_ylabel("")
    metric = "Mean PER score" if is_score else "PER response rate"
    test = "Mann-Whitney U" if is_score else "Fisher's exact"
    pairs = "pairwise, " if n > 2 else ""
    fig.suptitle(f"{title} — {metric}, {_comparison(groups, comparison)}\n"
                 f"{_n_text(summary, groups)} · unpaired between groups ({pairs}{test}, uncorrected)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    return fig


DELTA_YLIM = (-6, 6)


def build_delta_strip(paired, between, *, groups, trained_odor, title, comparison=None,
                      cohorts=(), concentrations=None):
    """Per-fly change in score (post - pre) per odor, the two groups side by side."""
    cohort = _phase_cohort(cohorts)
    after = f"after {phase_terms.block_label(cohort)}"
    odors = sorted(paired["odor"].unique(), key=str.casefold)
    x = np.arange(len(odors), dtype=float)
    n = len(groups)
    spacing = 0.8 / n
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(0.675 * n * max(len(odors), 1) + 2.2, 4.4))
    ax.axhline(0.0, color=INK_MUTED, linewidth=0.8, linestyle="--", zorder=0.5)
    handles = []
    positions = {}
    for gi, g in enumerate(groups):
        color = delta_colors(n)[gi]
        centres = x + (gi - (n - 1) / 2) * spacing
        positions[g] = centres
        for xc, odor in zip(centres, odors):
            sub = paired[(paired["group"] == g) & (paired["odor"] == odor)]
            d = (sub["score_post"] - sub["score_pre"]).to_numpy(dtype=float)
            if d.size == 0:
                continue
            jitter = rng.uniform(-0.15 * spacing, 0.15 * spacing, d.size)
            ax.scatter(xc + jitter, d, s=16, color=color, alpha=0.75, linewidths=0, zorder=3)
            m, e = float(np.mean(d)), _sem(d)
            ax.errorbar(xc + 0.25 * spacing, m, yerr=e, fmt="o", ms=5.5, color=color, mfc="white",
                        mew=1.4, elinewidth=1.2, capsize=2.5, zorder=4)
        _group_ticks(ax, centres, g)
        handles.append(plt.Line2D([], [], marker="o", ls="", color=color, label=g))
    ax.set_ylim(*DELTA_YLIM)
    ax.set_yticks(range(DELTA_YLIM[0], DELTA_YLIM[1] + 1, 2))
    ax.set_ylabel("Change in PER score (post − pre)")
    _recessive_grid(ax)
    _pair_brackets(ax, odors, positions, groups, between, "p_delta_score")
    ax.set_xticks(x)
    ax.set_xticklabels([odor_text(o, cohort, concentrations) for o in odors], rotation=28, ha="right")
    ax.tick_params(axis="x", pad=13, length=0)
    ax.set_xlim(-0.6, len(odors) - 0.4)
    _tint_trained(ax, odors, trained_odor)
    ax.set_title(f"{title} — change in PER score {after}, {_comparison(groups, comparison)}\n"
                 "dots = flies, open marker = mean ± SEM · Mann-Whitney U between groups "
                 f"({'pairwise, ' if n > 2 else ''}uncorrected)",
                 loc="left", color=INK, pad=8)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return fig


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def render_bar_figures(paired, summary, between, *, groups, trained_odor, title, out_dir,
                       comparison=None, tag="starvation", cohorts=(), concentrations=None):
    """``tag`` names what separates the groups in the file names (default "starvation").

    ``cohorts``: the datasets drawn, which pick the phase wording (``_phase_cohort``);
    ``concentrations`` (``odor_concentrations``) the percent in each odor name.
    """
    out_dir = Path(out_dir)
    written = []
    kw = dict(cohorts=cohorts, concentrations=concentrations)
    with plt.rc_context(_RC_CONTEXT):
        for kind in ("score", "rate"):
            written.append(_save(
                build_pre_post_bars(summary, groups=groups, kind=kind, trained_odor=trained_odor,
                                    title=title, comparison=comparison, **kw),
                out_dir / f"pre_vs_post_by_{tag}_{kind}.png"))
            written.append(_save(
                build_by_phase_bars(summary, between, groups=groups, kind=kind,
                                    trained_odor=trained_odor, title=title, comparison=comparison,
                                    **kw),
                out_dir / f"{tag}_compared_within_phase_{kind}.png"))
        written.append(_save(
            build_delta_strip(paired, between, groups=groups, trained_odor=trained_odor,
                              title=title, comparison=comparison, **kw),
            out_dir / "change_post_minus_pre_score.png"))
    return written


# ── Mean traces ──────────────────────────────────────────────────────────

TRACE_Y_LABEL = "Max Distance x Angle % (baseline mean subtracted)"


def group_traces(
    wide: pd.DataFrame, group: Group, *, trial_type: str, flies: set[tuple[str, int]],
    fps: float, odor_on_s: float,
) -> dict[str, np.ndarray]:
    """odor -> (n_flies, T) baseline-corrected traces for the flies in ``flies``."""
    from scripts.analysis.dataset_mean_traces_tvc import _prepare

    rows = select_group_rows(wide, group)
    rows = rows[[(str(f), int(n)) in flies
                 for f, n in zip(rows["fly"], rows["fly_number"])]]
    per_odor, _rows, _df = _prepare(rows, group.dataset, fps=fps, odor_on_s=odor_on_s,
                                    trial_type=trial_type)
    out = {}
    for odor, per_fly in per_odor.items():
        traces = [np.asarray(t, dtype=float) for t in per_fly.values() if np.size(t)]
        if not traces:
            continue
        width = max(t.size for t in traces)
        arr = np.full((len(traces), width), np.nan)
        for i, t in enumerate(traces):
            arr[i, : t.size] = t
        out[odor] = arr
    return out


def build_trace_grid(traces_by_group, *, groups, phase_index, odors, fps, odor_on_s,
                     odor_off_s, trained_odor, title, phase_label, comparison=None, cohorts=(),
                     concentrations=None):
    cohort = _phase_cohort(cohorts)
    odor_tag = phase_terms.odor_tag(cohort)
    ncols = 4
    nrows = int(np.ceil(len(odors) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.5 * nrows + 0.6),
                             sharey=True, squeeze=False)
    for ax in axes.flat[len(odors):]:
        ax.axis("off")
    for ax, odor in zip(axes.flat, odors):
        ax.axvspan(odor_on_s, odor_off_s, alpha=0.10, color="grey", linewidth=0)
        for gi, g in enumerate(groups):
            arr = traces_by_group[g].get(odor)
            if arr is None or arr.shape[0] == 0:
                continue
            with np.errstate(all="ignore"):
                mean = np.nanmean(arr, axis=0)
                sem = (np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
                       if arr.shape[0] > 1 else np.zeros_like(mean))
            t = np.arange(mean.size) / fps
            color = group_colors(len(groups))[gi][phase_index]
            ax.fill_between(t, mean - sem, mean + sem, color=color, alpha=0.22, linewidth=0)
            ax.plot(t, mean, color=color, linewidth=1.4, label=f"{g} (n={arr.shape[0]})")
        is_cs = bool(trained_odor) and odor.casefold().startswith(trained_odor.casefold())
        ax.set_title(odor_text(odor, cohort, concentrations) + (f"  ({odor_tag})" if is_cs else ""),
                     loc="left", fontsize=9,
                     color=odor_bar_palette.trained_tick_color(trained_odor) if is_cs else INK,
                     fontweight="bold" if is_cs else "normal")
        ax.legend(loc="upper right", fontsize=7)
        _recessive_grid(ax)
        ax.set_xlim(0, 90)
    for ax in axes[:, 0]:
        ax.set_ylabel("PER (Δ %)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)")
    fig.suptitle(f"{title} — mean PER trace, {phase_label}: {_comparison(groups, comparison)}\n"
                 f"mean ± SEM over flies; {TRACE_Y_LABEL}; grey = odor on",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    return fig


def render_trace_figures(paired, groups_spec, *, pretest_wide, testing_wide, fps, odor_on_s,
                         odor_off_s, trained_odor, title, out_dir, comparison=None,
                         tag="starvation", concentrations=None):
    out_dir = Path(out_dir)
    labels = [g.label for g in groups_spec]
    cohorts = [g.dataset for g in groups_spec]
    pre_name, post_name = _phase_names(_phase_cohort(cohorts))
    odors = sorted(paired["odor"].unique(), key=str.casefold)
    written = []
    for phase_index, (wide, trial_type, phase_label, stem) in enumerate((
        (pretest_wide, "pretest", pre_name, f"mean_traces_pretest_by_{tag}.png"),
        (testing_wide, "testing", post_name, f"mean_traces_posttraining_by_{tag}.png"),
    )):
        traces = {}
        for g in groups_spec:
            sub = paired[paired["group"] == g.label]
            flies = {(str(f), int(n)) for f, n in zip(sub["fly"], sub["fly_number"])}
            traces[g.label] = group_traces(wide, g, trial_type=trial_type, flies=flies,
                                           fps=fps, odor_on_s=odor_on_s)
        with plt.rc_context(_RC_CONTEXT):
            fig = build_trace_grid(traces, groups=labels, phase_index=phase_index, odors=odors,
                                   fps=fps, odor_on_s=odor_on_s, odor_off_s=odor_off_s,
                                   trained_odor=trained_odor, title=title,
                                   phase_label=phase_label, comparison=comparison,
                                   cohorts=cohorts, concentrations=concentrations)
            written.append(_save(fig, out_dir / stem))
    return written


# ── CLI ──────────────────────────────────────────────────────────────────


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--predictions-csv", required=True, type=Path)
    p.add_argument("--pretest-wide-csv", type=Path, default=None)
    p.add_argument("--testing-wide-csv", type=Path, default=None)
    p.add_argument("--group", action="append", required=True,
                   help="LABEL=DATASET[:FLY_PREFIX]; two or three, least starved first.")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--binary-threshold", type=int, default=BINARY_THRESHOLD)
    p.add_argument("--flagged-flies-csv", default="")
    p.add_argument("--trained-odor", default="Isoamyl Acetate")
    p.add_argument("--title", default="")
    p.add_argument("--fps", type=float, default=40.0)
    p.add_argument("--odor-on-s", type=float, default=30.0)
    p.add_argument("--odor-off-s", type=float, default=60.0)
    p.add_argument("--comparison", default="",
                   help='Title phrase for the two groups. Default: "<A> vs <B> starved".')
    p.add_argument("--file-tag", default="starvation",
                   help="What separates the groups, used in file names (e.g. 'day').")
    p.add_argument("--labels-csv", type=Path, default=None,
                   help="Predictions table with the trial labels as recorded, whose concentrations "
                        "name the odors on sensitivity figures. Default: --predictions-csv. Pass "
                        "the live table when --predictions-csv is an already-relabelled snapshot.")
    return p.parse_args(argv)


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    groups = [parse_group(s) for s in args.group]
    check_group_count(groups)
    labels = [g.label for g in groups]
    title = args.title or " vs ".join(labels)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # The relabelled subset goes to disk: load_paired_scores reads a path, and
    # the file doubles as the exact input record of these figures.
    raw = pd.read_csv(args.predictions_csv)
    preds = drop_flagged(subset_and_relabel(raw, groups), args.flagged_flies_csv)
    subset_csv = out_dir / "inputs" / "predictions_subset_relabelled.csv"
    subset_csv.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(subset_csv, index=False)

    paired = assign_groups(
        load_paired_scores(subset_csv, flagged_flies_csv=args.flagged_flies_csv), groups)
    if paired.empty:
        raise SystemExit("no paired pre/post rows for the requested groups")
    paired.to_csv(out_dir / "paired_scores_per_fly.csv", index=False)

    summary = group_summary(paired, labels, binary_threshold=args.binary_threshold)
    between = pairwise_between_stats(paired, labels, binary_threshold=args.binary_threshold)
    summary.to_csv(out_dir / "summary_by_group.csv", index=False)
    between.to_csv(out_dir / "stats_between_groups.csv", index=False)
    if len(labels) > 2:
        omnibus_stats(paired, labels, binary_threshold=args.binary_threshold).to_csv(
            out_dir / "stats_omnibus.csv", index=False)

    comparison = args.comparison or None
    # Display only: the tables above keep the normalised (concentration-free) odors.
    concentrations = odor_concentrations(raw if args.labels_csv is None else args.labels_csv,
                                         flies=paired)
    written = render_bar_figures(paired, summary, between, groups=labels,
                                 trained_odor=args.trained_odor, title=title, out_dir=out_dir,
                                 comparison=comparison, tag=args.file_tag,
                                 cohorts=[g.dataset for g in groups], concentrations=concentrations)
    if args.pretest_wide_csv and args.testing_wide_csv:
        written += render_trace_figures(
            paired, groups,
            pretest_wide=drop_flagged(subset_and_relabel(
                _read_table(args.pretest_wide_csv), groups), args.flagged_flies_csv),
            testing_wide=drop_flagged(subset_and_relabel(
                _read_table(args.testing_wide_csv), groups), args.flagged_flies_csv),
            fps=args.fps, odor_on_s=args.odor_on_s, odor_off_s=args.odor_off_s,
            trained_odor=args.trained_odor, title=title, out_dir=out_dir,
            comparison=comparison, tag=args.file_tag, concentrations=concentrations)
    for path in written:
        print(f"[OK] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
