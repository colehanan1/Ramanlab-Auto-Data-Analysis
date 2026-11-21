"""Figures for previous_odor_carryover.py (hand-run)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import previous_odor_carryover as poc  # noqa: E402

# Cohort slots match the folder's cross-cohort figures (dataviz reference palette,
# validated there); aqua/yellow sit under 3:1 on white, so cohorts also get markers.
COHORT_COLORS = dict(zip(poc.COHORTS, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")))
COHORT_MARKERS = dict(zip(poc.COHORTS, ("o", "s", "D", "^")))
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
# Diverging: blue <-> gray <-> red (reference diverging pair, neutral gray midpoint)
DIVERGING = LinearSegmentedColormap.from_list(
    "carryover", ["#174a8a", "#2a78d6", "#f0efec", "#e34948", "#9b2423"])
SHORT = {"Isoamyl Acetate": "IAA", "Ethyl Butyrate": "EB", "Hexanol": "Hex", "3-Octanol": "3-Oct",
         "Linalool": "Lin", "Citral": "Cit", "Benzaldehyde": "Benz", "Paraffin Oil": "Paraffin"}
PREV_SHORT = {**SHORT, "Paraffin Oil": "Paraffin\n(training)"}
COHORT_SHORT = {"IAA-Sensitivity-1-1": "IAA-exposed", "Hex-Sensitivity-1-0.1": "Hex-exposed",
                "3Oct-Sensitivity-1-0.1": "3-Oct-exposed", "Paraffin-Sensitivity-1-100": "Paraffin (control)"}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK_2, "ytick.color": INK_2, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def _matrix(cells: pd.DataFrame, value: str):
    M = np.full((len(poc.CURRENT_ODORS), len(poc.PREV_ODORS)), np.nan)
    N = np.zeros_like(M, dtype=int)
    for _, r in cells.iterrows():
        i, j = poc.CURRENT_ODORS.index(r.current_odor), poc.PREV_ODORS.index(r.previous_odor)
        M[i, j], N[i, j] = r[value], r.n_trials
    return M, N


def _heat(ax, cells, vmax, show_n=True, mark_p=True, ylabels=True, min_n=1):
    M, N = _matrix(cells, "adjusted_effect")
    M[N < min_n] = np.nan
    P, _ = _matrix(cells, "p")
    ax.imshow(np.ma.masked_invalid(M), cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isfinite(M[i, j]):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch="////",
                                           edgecolor=GRID, lw=0))
                continue
            dark = abs(M[i, j]) > 0.6 * vmax
            txt = f"{M[i, j]:+.0f}"
            if mark_p and P[i, j] < 0.05:
                txt += "*"
            ax.text(j, i - (0.12 if show_n else 0), txt, ha="center", va="center", fontsize=8,
                    color="white" if dark else INK)
            if show_n:
                ax.text(j, i + 0.27, f"n={N[i, j]}", ha="center", va="center", fontsize=5.5,
                        color="white" if dark else INK_2)
    ax.set_xticks(range(len(poc.PREV_ODORS)), [PREV_SHORT[o] for o in poc.PREV_ODORS])
    ax.set_yticks(range(len(poc.CURRENT_ODORS)),
                  [SHORT[o] for o in poc.CURRENT_ODORS] if ylabels else [])
    ax.set_xlabel("Previous trial's odor")
    if ylabels:
        ax.set_ylabel("This trial's odor")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


def _bh_note(cells: pd.DataFrame, scope: str) -> str:
    q = cells.loc[cells.cohort == scope, "q_bh_within_cohort"]
    n = int((q < 0.05).sum())
    return "none survives BH" if n == 0 else f"{n} survive BH (q < 0.05)"


PANEL_VIEWS = (("pre", "pre-exposure panel", "pooled|pre", "pre"),
               ("post", "post-exposure panel", "pooled|post", "post"),
               ("all", "pre + post combined", "pooled", "all"))


def _cbar(fig, cax, vmax, label):
    sm = plt.cm.ScalarMappable(cmap=DIVERGING, norm=plt.Normalize(-vmax, vmax))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(label)
    cb.outline.set_visible(False)


def matrix_three(out: Path, res: dict, vmax=30, fname="carryover_matrix_pre_post_combined.png",
                 title="the four Sensitivity cohorts",
                 note_extra=" Diagonal and paraffin column (post panel): post trial 1 straight after training "
                            "(90 s onset to onset; panel trials 330 s)."):
    """Pooled current x previous matrix for the pre panel, the post panel and both."""
    cells = res["responded"]["cells"]
    t = res["responded"]["tests"].set_index(["test", "scope"])
    fig = plt.figure(figsize=(19, 6.0))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.03], wspace=0.12)
    for k, (key, lab, scope, test_scope) in enumerate(PANEL_VIEWS):
        ax = fig.add_subplot(gs[k])
        _heat(ax, cells[cells.cohort == scope], vmax, ylabels=(k == 0), min_n=1)
        ax.set_title(f"{lab}\nany pair p = {t.loc[('pair', test_scope), 'p']:.2f} · previous-odor main p = "
                     f"{t.loc[('main', test_scope), 'p']:.2f} · {_bh_note(cells, scope)}",
                     loc="left", fontsize=9.5, color=INK)
    _cbar(fig, fig.add_subplot(gs[3]), vmax, "% responding vs this odor's average\n(percentage points, adjusted)")
    fig.suptitle(f"Does the previous odor shift PER to the next one? {title}", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.text(0.01, 0.01, "Cell: responses after that predecessor minus expectation (fly × panel, current odor × "
             "cohort × panel and position adjusted); n = trials. * raw p < 0.05 (batch-panel permutation). "
             "Hatched: no trials (an odor never follows itself within a panel)." + note_extra,
             fontsize=7, color=INK_2)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.82, bottom=0.14)
    fig.savefig(out / fname, dpi=170)
    plt.close(fig)


def matrix_by_cohort(out: Path, res: dict, panel: str = "all", vmax=50):
    cells = res["responded"]["cells"]
    tests = res["responded"]["tests"].set_index(["test", "scope"])
    lab = {"pre": "pre-exposure panel", "post": "post-exposure panel", "all": "pre + post combined"}[panel]
    fig = plt.figure(figsize=(11, 8.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.04], wspace=0.12, hspace=0.38)
    for k, c in enumerate(poc.COHORTS):
        scope = c if panel == "all" else f"{c}|{panel}"
        ax = fig.add_subplot(gs[k // 2, k % 2])
        _heat(ax, cells[cells.cohort == scope], vmax, show_n=True, ylabels=(k % 2 == 0), min_n=poc.MIN_CELL_N)
        ax.set_title(f"{COHORT_SHORT[c]}   pair p = {tests.loc[('pair', scope), 'p']:.2f}, "
                     f"main p = {tests.loc[('main', scope), 'p']:.2f}", loc="left", fontsize=9, color=INK)
    _cbar(fig, fig.add_subplot(gs[:, 2]), vmax, "% responding vs expectation (pp, adjusted)")
    con = tests.loc[("consistency", panel)]
    fig.suptitle(f"Previous-odor effect, each cohort separately ({lab})\n"
                 f"Same pattern across cohorts? mean between-cohort r = {con.observed:+.2f} "
                 f"(no shared pattern: ~0), p = {con.p:.2f}", x=0.02, ha="left", fontsize=10.5, color=INK)
    scopes = [c if panel == "all" else f"{c}|{panel}" for c in poc.COHORTS]
    nb = sum(int((cells.loc[cells.cohort == sc_, "q_bh_within_cohort"] < 0.05).sum()) for sc_ in scopes)
    fig.text(0.02, 0.012, "Hatched: no trials or fewer than 5 in that cell (older batches had no paraffin "
             f"trial). Diagonal / paraffin column: post trial 1 after training. * raw p < 0.05; "
             f"{'none survives' if nb == 0 else f'{nb} survive'} BH within its cohort.",
             fontsize=7, color=INK_2)
    fig.subplots_adjust(left=0.07, right=0.93, top=0.89, bottom=0.08)
    name = {"all": "pre_post_combined"}.get(panel, panel)
    fig.savefig(out / f"carryover_matrix_by_cohort_{name}.png", dpi=180)
    plt.close(fig)


def rate_by_previous(out: Path, tr: pd.DataFrame, res: dict, panel: str = "all"):
    cells = res["responded"]["cells"]
    lab = {"pre": "pre-exposure panel", "post": "post-exposure panel", "all": "pre + post combined"}[panel]
    scope = "pooled" if panel == "all" else f"pooled|{panel}"
    if panel != "all":
        tr = tr[tr["block"] == panel]
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.4), sharey=True)
    x = np.arange(len(poc.PREV_ODORS))
    for ax, co in zip(axes.ravel(), poc.CURRENT_ODORS):
        sub = tr[tr.odor == co]
        if sub.empty:
            ax.set_visible(False)
            continue
        rate = sub.groupby("prev_odor")["responded"].agg(["mean", "size"]).reindex(poc.PREV_ODORS)
        ax.bar(x, rate["mean"] * 100, width=0.72, color="#c3c2b7", edgecolor=SURFACE, lw=2, zorder=2)
        for k, c in enumerate(poc.COHORTS):
            rc = sub[sub.cohort == c].groupby("prev_odor")["responded"].agg(["mean", "size"]) \
                .reindex(poc.PREV_ODORS)
            ok = (rc["size"].fillna(0) >= 3).to_numpy()
            ax.scatter(x[ok] + (k - 1.5) * 0.14, rc["mean"][ok] * 100, s=16, color=COHORT_COLORS[c],
                       marker=COHORT_MARKERS[c], edgecolor=SURFACE, lw=0.6, zorder=3,
                       label=COHORT_SHORT[c])
        overall = sub["responded"].mean() * 100
        ax.axhline(overall, color=INK_2, lw=1, ls="--", zorder=1)
        for j, o in enumerate(poc.PREV_ODORS):
            if np.isfinite(rate.loc[o, "mean"]):
                ax.text(j, 1.5, f"{int(rate.loc[o, 'size'])}", ha="center", fontsize=6, color=INK_2, zorder=4)
        pc = cells[(cells.cohort == scope) & (cells.current_odor == co)]
        ax.set_title(f"{co}  (all: {overall:.0f} %)", loc="left", fontsize=9, color=INK)
        ax.set_xticks(x, [SHORT[o] if o != "Paraffin Oil" else "Par.(tr)" for o in poc.PREV_ODORS],
                      fontsize=7)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        best = pc.sort_values("p").iloc[0] if len(pc) else None
        if best is not None:
            ax.text(0.98, 0.97, f"lowest cell p {best.p:.2f}\n(after {SHORT[best.previous_odor]})",
                    transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=INK_2)
    for ax in axes[:, 0]:
        ax.set_ylabel("% responding (score ≥ 2)")
    for ax in axes[1]:
        ax.set_xlabel("Previous trial's odor")
    axes[0, 0].set_ylim(0, 100)
    h, lab_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab_, loc="upper right", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle(f"Raw % responding to each odor, split by the odor on the trial before ({lab})\n"
                 "grey bar: all cohorts pooled (number = trials); markers: each cohort (≥ 3 trials); "
                 "dashed: that odor's overall rate; post trial 1 counts its training odor as the one before",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.subplots_adjust(top=0.86)
    name = {"all": "pre_post_combined"}.get(panel, panel)
    fig.savefig(out / f"rate_by_previous_odor_{name}.png", dpi=180)
    plt.close(fig)


def directed_forest(out: Path, res: dict):
    t = res["responded"]["tests"]
    names = {"same_class": "Previous odor same chemical class\n(e.g. IAA after EB, Hex after 3-Oct)",
             "attractiveness": "Previous odor more attractive\n(per +10 pp of its own response rate)",
             "after_sensitized": "Previous odor = the cohort's\nsensitized odor (exposed cohorts)"}
    scopes = [("all", "Pooled"), ("pre", "Pooled, pre-exposure panel"), ("post", "Pooled, post-exposure panel"),
              ("post_trial1_after_training", "Post trial 1 (after training)"),
              ("panel_transitions_only", "Panel transitions only")] \
        + [(c, COHORT_SHORT[c]) for c in poc.COHORTS]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.0), sharey=True)
    for ax, (kind, title) in zip(axes, names.items()):
        sub = t[t.test == kind].set_index("scope")
        scale = 10 if kind == "attractiveness" else 1
        for i, (s, lab) in enumerate(scopes):
            if s not in sub.index:
                continue
            r = sub.loc[s]
            col = COHORT_COLORS.get(s, INK)
            mk = COHORT_MARKERS.get(s, "o")
            ax.plot([r.ci_low * scale, r.ci_high * scale], [i, i], color=col, lw=2, solid_capstyle="round")
            ax.scatter(r.observed * scale, i, s=40, color=col, marker=mk, edgecolor=SURFACE, lw=1, zorder=3)
            ax.text(1.01, i, f"p {r.p:.2f}", transform=ax.get_yaxis_transform(), va="center",
                    fontsize=7, color=INK_2)
        ax.axvline(0, color=INK_2, lw=1)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, loc="left", fontsize=9, color=INK)
        ax.set_xlabel("Shift in % responding (pp)")
        ax.grid(axis="x", color=GRID, lw=0.6)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    axes[0].set_yticks(range(len(scopes)), [lab for _, lab in scopes])
    axes[0].invert_yaxis()
    fig.suptitle("Directed carry-over contrasts (95 % band from the batch-panel permutation null; p uncorrected)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 0.98, 0.95), w_pad=4)
    fig.savefig(out / "directed_contrasts.png", dpi=180)
    plt.close(fig)


def after_training_figure(out: Path, at: pd.DataFrame):
    """Post trial 1 (the trained odor right before it) by first-test odor and cohort."""
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9), sharey=True)
    x = np.arange(len(poc.RANDOMIZED_ODORS))
    for ax, c in zip(axes, poc.COHORTS):
        sub = at[(at.cohort == c) & (at.first_test_odor != "(all odors)")].set_index("first_test_odor") \
            .reindex(poc.RANDOMIZED_ODORS)
        allr = at[(at.cohort == c) & (at.first_test_odor == "(all odors)")].iloc[0]
        ax.bar(x, sub["raw_trial1"], width=0.7, color=COHORT_COLORS[c], edgecolor=SURFACE, lw=2, zorder=2,
               label="post trial 1 (straight after training)")
        ax.scatter(x, sub["raw_same_odor_post_trials_2plus"], s=34, facecolor=SURFACE, edgecolor=INK,
                   lw=1.2, zorder=3, label="same odor, post trials 2+")
        for j, o in enumerate(poc.RANDOMIZED_ODORS):
            n = sub.loc[o, "n_flies"]
            if np.isfinite(n):
                ax.text(j, 2, f"{int(n)}", ha="center", fontsize=6.5, color=INK_2, zorder=4)
            if o == allr.trained_odor:
                ax.text(j, 96, "trained", ha="center", fontsize=6.5, color=INK, rotation=0)
        ax.set_xticks(x, [SHORT[o] for o in poc.RANDOMIZED_ODORS], fontsize=7.5)
        ax.set_title(f"{COHORT_SHORT[c]}: trained {SHORT.get(allr.trained_odor, allr.trained_odor)}\n"
                     f"all first-test odors: {allr.adjusted_effect:+.1f} pp adj. "
                     f"[{allr.ci_low:+.0f}, {allr.ci_high:+.0f}]", loc="left", fontsize=8.5, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xlabel("Odor on post trial 1")
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("% responding (score ≥ 2)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=2, frameon=False, fontsize=8)
    fig.suptitle("The trained odor as the 'previous odor': first post-exposure trial, by which odor it was "
                 "(numbers = flies; few batches each, so read the pooled tests, not single bars)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.subplots_adjust(top=0.76)
    fig.savefig(out / "after_training_first_test.png", dpi=180)
    plt.close(fig)


def focus_forest(out: Path, focus: pd.DataFrame):
    """The pairs looked at closely: rows = panel (pre, post, pre+post), columns = pair."""
    f = focus[focus.outcome == "responded"]
    pairs = list(dict.fromkeys(f["pair"]))
    panels = ["pre", "post", "pre+post"]
    order = ["Sensitivity cohorts (pooled)"] + poc.COHORTS + ["Other datasets (pooled)", "All datasets combined"]
    labels = {"Sensitivity cohorts (pooled)": "Sensitivity, pooled", "Other datasets (pooled)":
              "Other datasets (replication)", "All datasets combined": "All datasets combined", **COHORT_SHORT}
    fig, axes = plt.subplots(len(panels), len(pairs), figsize=(4.3 * len(pairs), 3.3 * len(panels)),
                             sharey=True, squeeze=False)
    for r_, panel in enumerate(panels):
        for c_, pr in enumerate(pairs):
            ax = axes[r_, c_]
            sub = f[(f.pair == pr) & (f.panel == panel)].set_index("data")
            for i, d in enumerate(order):
                if d not in sub.index or not np.isfinite(sub.loc[d].get("cluster_estimate", np.nan)):
                    continue
                r = sub.loc[d]
                few = r.n_batches_with_pair < 10
                col = COHORT_COLORS.get(d, INK)
                ax.plot([r.cluster_ci_low, r.cluster_ci_high], [i, i], color=MUTED if few else col, lw=2,
                        ls=(0, (2, 1.5)) if few else "-", solid_capstyle="round", clip_on=True)
                ax.scatter(r.cluster_estimate, i, s=38, color=col, marker=COHORT_MARKERS.get(d, "o"),
                           edgecolor=SURFACE, lw=1, zorder=3)
                pp = f"{r.perm_p:.3f}" if np.isfinite(r.perm_p) else "–"
                ax.text(1.01, i, f"{int(r.n_trials)}/{int(r.n_batches_with_pair)}b p {pp}",
                        transform=ax.get_yaxis_transform(), va="center", fontsize=6, color=INK_2)
            ax.axvline(0, color=INK_2, lw=1)
            ax.axhline(4.5, color=GRID, lw=1)
            ax.set_xlim(-60, 60)
            if r_ == 0:
                ax.set_title(pr.replace(" -> ", " → "), loc="left", fontsize=10, color=INK)
            if c_ == 0:
                ax.set_ylabel(panel + (" panel" if panel != "pre+post" else " combined"), fontsize=10, color=INK)
            if r_ == len(panels) - 1:
                ax.set_xlabel("Shift in % responding (pp)")
            ax.grid(axis="x", color=GRID, lw=0.6)
            ax.tick_params(axis="y", length=0)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
    axes[0, 0].set_yticks(range(len(order)), [labels[d] for d in order], fontsize=7.5)
    axes[0, 0].invert_yaxis()
    fig.suptitle("Pairs looked at closely (the odor before → this odor), pre panel, post panel and both. "
                 "Dot + line: batch-clustered regression, 95 % CI; dashed grey: pair seen in < 10 batches "
                 "(CI unreliable, use the permutation p). Text: trials / batches, permutation p.",
                 x=0.01, ha="left", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 0.97, 0.95), w_pad=5, h_pad=1.5)
    fig.savefig(out / "focus_pairs_pre_post_combined.png", dpi=170)
    plt.close(fig)


ODOR_COL, TRIAL_COL = "#2a78d6", "#eb6834"     # categorical slots 1 and 2


def trial_number_figure(out: Path, res: pd.DataFrame, rates: pd.DataFrame):
    rows_ = [("all cohorts", "Sensitivity pooled")] + [(c, COHORT_SHORT[c]) for c in poc.COHORTS] \
        + [("other datasets (pooled)", "Other datasets")]
    res = res.set_index("scope")
    fig = plt.figure(figsize=(16, 9.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.45, wspace=0.75)
    for k, (blk, lab) in enumerate((("pre", "Pre-exposure panel"), ("post", "Post-exposure panel"),
                                    ("pre+post", "Pre + post combined"))):
        ax = fig.add_subplot(gs[0, k])
        keys = [f"{r}|{blk}" for r, _ in rows_]
        sub = res.reindex(keys)
        y = np.arange(len(keys))
        ax.barh(y - 0.19, sub["heldout_r2_odor_pct"], height=0.36, color=ODOR_COL, label="odor delivered", zorder=2)
        ax.barh(y + 0.19, sub["heldout_r2_trial_number_pct"], height=0.36, color=TRIAL_COL,
                label="trial number", zorder=2)
        for i, (_, r) in enumerate(sub.iterrows()):
            if np.isfinite(r.get("p_odor_beyond_trial_number", np.nan)):
                ax.text(1.01, i, f"odor p {r.p_odor_beyond_trial_number:.3f}\ntrial no. p "
                        f"{r.p_trial_number_beyond_odor:.3f}", transform=ax.get_yaxis_transform(),
                        va="center", fontsize=6.3, color=INK_2)
        ax.set_yticks(y, [l for _, l in rows_] if k == 0 else [], fontsize=8.5)
        ax.invert_yaxis()
        ax.axvline(0, color=INK_2, lw=1)
        ax.axhline(4.5, color=GRID, lw=1)
        ax.set_xlim(-7, 10)
        ax.set_xlabel("Within-fly variance in PER explained,\nheld-out batches (%)")
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(lab, loc="left", fontsize=10.5, color=INK)
        if k == 0:
            ax.legend(loc="lower right", frameon=False, fontsize=8)
    allr = rates[rates.cohort == "all"]
    for k, (fac, title) in enumerate((("odor", "% responding by odor"), ("trial_number", "% responding by trial number"))):
        a = fig.add_subplot(gs[1, k if k == 0 else 1:3] if k == 1 else gs[1, 0])
        sub = allr[allr.factor == fac]
        levels = poc.RANDOMIZED_ODORS if fac == "odor" else sorted(sub["level"].unique(), key=float)
        xx = np.arange(len(levels))
        for j, (b_, col) in enumerate((("pre", "#c3c2b7"), ("post", INK_2))):
            v = sub[sub.block == b_].set_index("level").reindex(levels)["pct_responding"]
            a.bar(xx + (j - 0.5) * 0.38, v, width=0.36, color=col, label=f"{b_}-exposure panel", zorder=2)
        a.set_xticks(xx, [SHORT.get(l, l) for l in levels] if fac == "odor"
                     else [str(int(float(l))) for l in levels], fontsize=8)
        a.set_title(f"Sensitivity pooled: {title}", loc="left", fontsize=9.5,
                    color=ODOR_COL if fac == "odor" else TRIAL_COL)
        a.set_ylim(0, 60)
        a.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
        a.set_xlabel("Odor" if fac == "odor" else "Trial number in the panel")
        a.set_ylabel("% responding (score ≥ 2)")
    fig.suptitle("Does the trial number predict PER better than the odor delivered? (7 randomized odors)\n"
                 "p: exact randomization tests, trial numbers or odors shuffled within each batch panel; "
                 "< 0 % = predicts held-out flies worse than their own average",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.07)
    fig.savefig(out / "trial_number_vs_odor_pre_post_combined.png", dpi=170)
    plt.close(fig)


def draw_all(out: Path, tr: pd.DataFrame, res: dict, ext: dict | None = None, trials=None):
    matrix_three(out, res)
    for panel in ("pre", "post", "all"):
        matrix_by_cohort(out, res, panel)
        rate_by_previous(out, tr, res, panel)
    directed_forest(out, res)
    after_training_figure(out, res["responded"]["after_training"])
    if "focus" in res:
        focus_forest(out, res["focus"])
    if ext is not None:
        info = ext["info"]
        matrix_three(out / "Other-Datasets", ext, vmax=15, fname="carryover_matrix_other_datasets_pre_post_combined.png",
                     title=f"other datasets ({info['datasets']} datasets, {info['flies']} flies, "
                           f"{info['batches']} batches)", note_extra=" Panel transitions only.")
    comb = res.get("combined")
    if comb is not None:
        info = comb["info"]
        matrix_three(out / "All-Datasets-Combined", comb, vmax=15,
                     fname="carryover_matrix_all_datasets_pre_post_combined.png",
                     title=f"all datasets combined ({info['datasets']} datasets, {info['flies']} flies, "
                           f"{info['batches']} batches)", note_extra=" Panel transitions only.")
