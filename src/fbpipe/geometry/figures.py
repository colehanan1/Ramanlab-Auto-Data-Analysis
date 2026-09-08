"""Figures and the per-trial summary of the geometry pipeline.

Per fly: (1) PER % and abdomen angle % on ONE shared 0-100 axis (never a twin
axis), one row per trial in protocol order, the trial's own odor window shaded
(light window for LightOnly trials); (2) abdomen length/width as % of rest,
same layout; per batch: (3) a QC grid of tracked fractions and (4)
``<batch>_trial_summary.csv`` with per-trial window metrics -- per trial only,
never an average of averages.

Colours follow the lab's validated categorical order (dataviz reference
palette, slots 1-2: blue, orange; CVD-validated 2026-09-08). Text never wears a
series colour.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from ..config import Settings
from ..plot_style import apply_lab_style
from . import columns as C
from .config import GeometrySettings
from .steps import load_trial_entries
from .trial_meta import TrialInfo

apply_lab_style()

SERIES_1 = "#2a78d6"          # categorical slot 1 (blue)
SERIES_2 = "#eb6834"          # categorical slot 2 (orange)
PER_COLOR = SERIES_1
ANGLE_COLOR = SERIES_2
LENGTH_COLOR = SERIES_1
WIDTH_COLOR = SERIES_2
TEXT_SECONDARY = "#52514e"
ODOR_SPAN_COLOR = "#9e9e9e"
ODOR_SPAN_ALPHA = 0.20
LIGHT_LINE_COLOR = "#e8a000"
LIGHT_SPAN_COLOR = "#ffd24d"
LIGHT_SPAN_ALPHA = 0.30
TRAINED_TITLE_COLOR = "tab:blue"
TIME_TICK_S = 15.0


@dataclass
class TrialPanel:
    info: TrialInfo
    t: np.ndarray
    per: np.ndarray
    angle: np.ndarray
    length: np.ndarray
    width: np.ndarray
    status: pd.Series


def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    return df[name].to_numpy(dtype=float) if name in df.columns else np.full(len(df), np.nan)


def load_panels(batch_dir: Path, gcfg: GeometrySettings) -> Dict[int, List[TrialPanel]]:
    panels: Dict[int, List[TrialPanel]] = {}
    for _, info, tables in load_trial_entries(batch_dir, gcfg):
        for slot, path in sorted(tables.items()):
            df = C.read_table_any(path)
            panels.setdefault(slot, []).append(TrialPanel(
                info=info, t=_col(df, C.TIMESTAMP), per=_col(df, C.PRO_PCT), angle=_col(df, C.ABD_ANGLE_PCT),
                length=_col(df, C.ABD_LEN_PCT), width=_col(df, C.ABD_WID_PCT),
                status=df[C.STATUS].astype(str) if C.STATUS in df.columns else pd.Series([], dtype=str)))
    return panels


def trained_odor(panels: Sequence[TrialPanel]) -> Optional[str]:
    odors = [p.info.odor for p in panels if p.info.trial_type == "training" and p.info.odor]
    return max(set(odors), key=odors.count) if odors else None


def _stimulus(ax, info: TrialInfo) -> None:
    if info.has_odor and info.odor_on_s is not None and info.odor_off_s is not None:
        ax.axvspan(info.odor_on_s, info.odor_off_s, color=ODOR_SPAN_COLOR, alpha=ODOR_SPAN_ALPHA, lw=0, zorder=0)
    elif info.light_on_s is not None:
        off = info.light_off_s if info.light_off_s is not None else info.duration_s
        ax.axvspan(info.light_on_s, off, color=LIGHT_SPAN_COLOR, alpha=LIGHT_SPAN_ALPHA, lw=0, zorder=0)
        ax.axvline(info.light_on_s, linestyle="-.", lw=1.2, color=LIGHT_LINE_COLOR, zorder=1)


def _dominant_status(status: pd.Series) -> str:
    bad = status[status != C.STATUS_OK]
    return str(bad.mode().iloc[0]) if len(bad) else C.STATUS_OK


def _panel_title(ax, info: TrialInfo, trained: Optional[str]) -> None:
    label = f"{info.trial_type} {info.trial_index}: {info.odor or ''}".strip()
    color = TRAINED_TITLE_COLOR if trained and info.odor == trained else "black"
    ax.set_title(label, loc="left", fontsize=9, weight="bold", color=color, pad=2)


Series = Callable[[TrialPanel], Tuple[Tuple[np.ndarray, str, str], ...]]


def _y_top(panels: Sequence[TrialPanel], series: Series, floor: float = 100.0, cap: float = 250.0) -> float:
    """Upper axis limit for percentages that may exceed 100 (box size vs rest): the 99.5th
    percentile of the data rounded up to the next 10, never below ``floor`` nor above ``cap``."""
    vals = np.concatenate([np.asarray(arr, dtype=float).ravel() for p in panels for arr, _, _ in series(p)] or [np.array([])])
    vals = vals[np.isfinite(vals)]
    if not vals.size:
        return floor
    return float(min(cap, max(floor, 10.0 * np.ceil(np.percentile(vals, 99.5) / 10.0))))


def _stacked(panels: Sequence[TrialPanel], slot: int, dataset: str, batch_name: str, out_path: Path, *,
             trace_max_s: float, trained: Optional[str], series: Series, ylabel: str, title: str,
             y_top: float = 100.0) -> Path:
    n = max(1, len(panels))
    header_in = 1.25
    fig_h = max(3.4, n * 1.6 + header_in + 0.35)
    fig, axes = plt.subplots(n, 1, figsize=(10, fig_h), sharex=True, squeeze=False)
    axes = axes[:, 0]
    handles: Dict[str, object] = {}
    for ax, panel in zip(axes, panels):
        _stimulus(ax, panel.info)
        missing = []
        for arr, color, name in series(panel):
            if np.isfinite(arr).any():
                h, = ax.plot(panel.t, arr, lw=1.2, color=color, label=name, zorder=3)
                handles.setdefault(name, h)
            else:
                missing.append(name)
        if missing:
            ax.text(0.99, 0.88, f"{', '.join(missing)}: not trackable ({_dominant_status(panel.status)[:60]})",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7, color=TEXT_SECONDARY)
        _panel_title(ax, panel.info, trained)
        ax.set_ylim(0, y_top)
        ax.set_yticks([0, 50, 100] if y_top <= 100 else [0, 50, 100, y_top])
        ax.set_xlim(0, trace_max_s)
        ax.xaxis.set_major_locator(MultipleLocator(TIME_TICK_S if trace_max_s > 30 else max(trace_max_s / 3, 0.5)))
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Time (s)", fontsize=11)
    fig.text(0.02, 0.5, ylabel, va="center", rotation="vertical", fontsize=10)
    if handles:
        fig.legend(handles=list(handles.values()), labels=list(handles.keys()), loc="upper right",
                   bbox_to_anchor=(0.98, 1.0 - 0.15 / fig_h), fontsize=8, frameon=True)
    fig.tight_layout(rect=[0.04, 0, 1, 1.0 - header_in / fig_h])
    fig.suptitle(title, y=1.0 - 0.30 / fig_h, fontsize=14, weight="bold")
    fig.text(0.5, 1.0 - 0.80 / fig_h, f"{dataset}  -  {batch_name}  -  fly {slot}", ha="center", fontsize=12, weight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fly_traces(panels, slot, dataset, batch_name, out_path, *, trace_max_s, trained) -> Path:
    def series(p: TrialPanel):
        return ((p.per, PER_COLOR, "proboscis extension %"), (p.angle, ANGLE_COLOR, "abdomen angle %"))
    return _stacked(panels, slot, dataset, batch_name, Path(out_path), trace_max_s=trace_max_s, trained=trained,
                    series=series,
                    ylabel="% of fly range: proboscis (0 = min, 100 = max extension), abdomen angle (0 = straightest, 100 = most curled)",
                    title="Proboscis extension and abdomen angle across trials")


def plot_size_traces(panels, slot, dataset, batch_name, out_path, *, trace_max_s, trained) -> Path:
    def series(p: TrialPanel):
        return ((p.length, LENGTH_COLOR, "abdomen length % of rest"), (p.width, WIDTH_COLOR, "abdomen width % of rest"))
    return _stacked(panels, slot, dataset, batch_name, Path(out_path), trace_max_s=trace_max_s, trained=trained,
                    series=series, ylabel="Abdomen box size, % of pre-odor resting median",
                    title="Abdomen box size across trials", y_top=_y_top(panels, series))


_trapezoid = getattr(np, "trapezoid", None) or np.trapz   # numpy 2 renamed trapz


def _window_metrics(t: np.ndarray, y: np.ndarray, window) -> dict:
    if window is None:
        return {"max": np.nan, "auc": np.nan, "mean": np.nan}
    on, off = window
    m = (t >= on) & (t <= off) & np.isfinite(y)
    if not m.any():
        return {"max": np.nan, "auc": np.nan, "mean": np.nan}
    auc = float(_trapezoid(y[m], t[m])) if m.sum() > 1 else 0.0
    return {"max": float(np.nanmax(y[m])), "auc": auc, "mean": float(np.nanmean(y[m]))}


def trial_summary(panels_by_slot: Dict[int, List[TrialPanel]], dataset: str, batch_name: str) -> pd.DataFrame:
    rows = []
    for slot, panels in sorted(panels_by_slot.items()):
        for p in panels:
            win = p.info.stimulus_window_s
            per = _window_metrics(p.t, p.per, win)
            ang = _window_metrics(p.t, p.angle, win)
            length = _window_metrics(p.t, p.length, win)
            width = _window_metrics(p.t, p.width, win)
            n = int(p.t.size)
            rows.append({
                "dataset": dataset, "fly": batch_name, "fly_number": slot, "trial_label": p.info.label,
                "trial_type": p.info.trial_type, "trial_index": p.info.trial_index, "odor": p.info.odor,
                "has_odor": bool(p.info.has_odor),
                "window_on_s": None if win is None else win[0], "window_off_s": None if win is None else win[1],
                "n_frames": n,
                "per_tracked_frac": float(np.isfinite(p.per).mean()) if n else 0.0,
                "abdomen_tracked_frac": float(np.isfinite(p.length).mean()) if n else 0.0,
                "angle_tracked_frac": float(np.isfinite(p.angle).mean()) if n else 0.0,
                "per_max_pct_window": per["max"], "per_auc_pct_s_window": per["auc"],
                "angle_max_pct_window": ang["max"], "angle_auc_pct_s_window": ang["auc"],
                "length_mean_pct_window": length["mean"], "width_mean_pct_window": width["mean"],
                "dominant_status": _dominant_status(p.status),
            })
    return pd.DataFrame(rows)


def plot_qc(panels_by_slot: Dict[int, List[TrialPanel]], dataset: str, batch_name: str, out_path: Path) -> Path:
    slots = sorted(panels_by_slot)
    labels: List[str] = []
    for slot in slots:
        for p in panels_by_slot[slot]:
            if p.info.label not in labels:
                labels.append(p.info.label)
    metrics = [("proboscis", lambda p: float(np.isfinite(p.per).mean())),
               ("abdomen box", lambda p: float(np.isfinite(p.length).mean())),
               ("abdomen angle", lambda p: float(np.isfinite(p.angle).mean()))]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 0.35 * max(len(labels), 4) + 1.8),
                             squeeze=False)
    for ax, (name, fn) in zip(axes[0], metrics):
        grid = np.full((len(labels), len(slots)), np.nan)
        for j, slot in enumerate(slots):
            by_label = {p.info.label: p for p in panels_by_slot[slot]}
            for i, lab in enumerate(labels):
                if lab in by_label:
                    grid[i, j] = fn(by_label[lab])
        ax.imshow(grid, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        for i in range(len(labels)):
            for j in range(len(slots)):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{100 * grid[i, j]:.0f}%", ha="center", va="center", fontsize=7,
                            color="white" if grid[i, j] > 0.6 else "black")
        ax.set_xticks(range(len(slots)))
        ax.set_xticklabels([f"fly {s}" for s in slots], fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(f"{name}: tracked frames", fontsize=9, loc="left")
    fig.suptitle(f"{dataset} - {batch_name}: fraction of frames tracked", fontsize=11, weight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate(batch_dir: Path, dataset: str, settings: Settings, gcfg: GeometrySettings, out_root: Path) -> dict:
    batch_dir = Path(batch_dir)
    panels_by_slot = load_panels(batch_dir, gcfg)
    out_dir = Path(out_root) / dataset / batch_dir.name
    figures: List[Path] = []
    if not panels_by_slot:
        return {"n_figures": 0, "figures": figures, "summary_csv": None, "out_dir": out_dir}
    for slot, panels in sorted(panels_by_slot.items()):
        trained = trained_odor(panels)
        figures.append(plot_fly_traces(panels, slot, dataset, batch_dir.name,
                                       out_dir / f"{batch_dir.name}_fly{slot}_per_abdomen_traces.png",
                                       trace_max_s=gcfg.trace_max_s, trained=trained))
        figures.append(plot_size_traces(panels, slot, dataset, batch_dir.name,
                                        out_dir / f"{batch_dir.name}_fly{slot}_abdomen_size_traces.png",
                                        trace_max_s=gcfg.trace_max_s, trained=trained))
    figures.append(plot_qc(panels_by_slot, dataset, batch_dir.name, out_dir / f"{batch_dir.name}_geometry_qc.png"))
    summary = trial_summary(panels_by_slot, dataset, batch_dir.name)
    summary_csv = out_dir / f"{batch_dir.name}_trial_summary.csv"
    summary.to_csv(summary_csv, index=False)
    return {"n_figures": len(figures), "figures": figures, "summary_csv": summary_csv, "out_dir": out_dir}
