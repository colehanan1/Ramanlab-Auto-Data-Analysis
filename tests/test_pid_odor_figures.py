"""Tests for the PID odor-delivery methods figures.

The golden numbers below come from the archived per-exposure latency exports
that the thesis text was written from:

* ``data_odor opto/all_odors/*_latency_rms.csv``   (optogenetic enclosure)
* ``odor data PID Manual/more/all_odors/aggregate_all_exposures.csv`` (manual)

The figure code must reproduce those exposures exactly -- it re-plots them, it
does not re-detect them.  The single-trace panel *is* re-detected from raw
voltage, so its onsets are pinned here too (read off the raw samples: the
benzaldehyde session crosses 3 sigma at +0.4 s on exposure 1 and +1.0 s on
exposure 2).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.analysis.pid_odor_figures import (
    BENZ_SESSION,
    EXCLUDED_EXPOSURES,
    ODOR_NAMES,
    detect_onset,
    exposure_windows,
    load_latency_table,
    load_pid_session,
    make_latency_figure,
    make_trace_figure,
    pooled_stats,
)


# --------------------------------------------------------------------------- #
# Aggregate latency dataset
# --------------------------------------------------------------------------- #
def test_latency_table_shape() -> None:
    """65 attempted exposures: 7 opto sessions + 6 manual sessions, 5 repeats each."""
    table = load_latency_table()
    assert len(table) == 65
    assert set(table["rig"]) == {"optogenetic", "manual"}
    assert table.groupby("rig")["session"].nunique().to_dict() == {"optogenetic": 7, "manual": 6}


def test_opto_stats_match_archive() -> None:
    """Optogenetic enclosure: n=28, 2.15 +/- 0.21 s -- the '2.2 +/- 0.2 s' in the text."""
    table = load_latency_table()
    opto = table[(table.rig == "optogenetic") & table.included]
    n, mean, sem = pooled_stats(opto.latency_s)
    assert n == 28
    assert mean == pytest.approx(2.153213, abs=1e-4)
    assert sem == pytest.approx(0.213281, abs=1e-4)


def test_manual_stats_before_and_after_exclusion() -> None:
    """Manual enclosure: 27 detections; 25 after the two long exposures are dropped."""
    table = load_latency_table()
    manual = table[(table.rig == "manual") & table.latency_s.notna()]
    n_all, mean_all, sem_all = pooled_stats(manual.latency_s)
    assert (n_all, round(mean_all, 4), round(sem_all, 4)) == (27, 3.0124, 0.2569)

    kept = manual[manual.included]
    n, mean, sem = pooled_stats(kept.latency_s)
    assert n == 25
    assert mean == pytest.approx(2.764662, abs=1e-4)  # the '2.8 +/- 0.2 s' in the text
    assert sem == pytest.approx(0.196602, abs=1e-4)


def test_excluded_exposures_are_the_two_longest_manual_detections() -> None:
    table = load_latency_table()
    excluded = table[table.latency_s.notna() & ~table.included]
    assert len(excluded) == 2
    assert sorted(round(v, 3) for v in excluded.latency_s) == [5.008, 7.211]
    assert set(zip(excluded.session, excluded.repeat)) == set(EXCLUDED_EXPOSURES)
    assert set(excluded.rig) == {"manual"}


def test_pooled_stats_match_thesis_sentence() -> None:
    """Pooled across both enclosures: n=53, 2.44 +/- 0.15 s."""
    table = load_latency_table()
    n, mean, sem = pooled_stats(table[table.included].latency_s)
    assert n == 53
    assert mean == pytest.approx(2.4424, abs=1e-3)
    assert sem == pytest.approx(0.1513, abs=1e-3)


def test_opto_hexanol_has_no_detections() -> None:
    """The one channel that never deflected -- the figure must show it as n.d."""
    table = load_latency_table()
    hexanol = table[(table.rig == "optogenetic") & (table.odor_code == "H")]
    assert len(hexanol) == 5
    assert hexanol.latency_s.isna().all()


def test_odor_names_cover_every_code() -> None:
    table = load_latency_table()
    assert set(table.odor_code) <= set(ODOR_NAMES)
    assert ODOR_NAMES["B"] == "Benzaldehyde"
    assert ODOR_NAMES["O"] == "3-Octanol"


# --------------------------------------------------------------------------- #
# Single-trace dataset
# --------------------------------------------------------------------------- #
def test_benzaldehyde_exposure_windows() -> None:
    df = load_pid_session(BENZ_SESSION)
    windows = exposure_windows(df)
    assert len(windows) == 2
    assert windows[0].t_open == pytest.approx(40.05, abs=0.05)
    assert windows[0].t_close == pytest.approx(219.91, abs=0.05)
    assert windows[1].t_open == pytest.approx(430.21, abs=0.05)
    assert windows[1].t_close == pytest.approx(610.07, abs=0.05)


def test_benzaldehyde_onsets_and_baselines() -> None:
    df = load_pid_session(BENZ_SESSION)
    onsets = [detect_onset(df, w) for w in exposure_windows(df)]

    assert onsets[0].latency_s == pytest.approx(0.40, abs=0.05)
    assert onsets[1].latency_s == pytest.approx(1.00, abs=0.05)

    assert onsets[0].baseline_v == pytest.approx(0.6863, abs=1e-3)
    assert onsets[1].baseline_v == pytest.approx(0.6194, abs=1e-3)

    # Threshold sits above baseline and below the plume plateau in both exposures.
    for onset in onsets:
        assert onset.baseline_v < onset.threshold_v < onset.peak_v


def test_plateau_is_sustained_for_the_whole_presentation() -> None:
    """Panel b's claim: once the bolus arrives the signal holds for the full window."""
    from scripts.analysis.pid_odor_figures import OPTO_DIR, plateau_stats

    eb = load_pid_session(OPTO_DIR / "odor_test_1757376621.csv")
    window = exposure_windows(eb)[0]
    onset = detect_onset(eb, window)
    mean, sd = plateau_stats(eb, window, onset)
    assert mean == pytest.approx(0.7371, abs=1e-3)
    assert sd == pytest.approx(0.0292, abs=1e-3)
    assert sd / mean < 0.10  # flat to within 10% of the plateau height

    benz = load_pid_session(BENZ_SESSION)
    window = exposure_windows(benz)[0]
    mean, sd = plateau_stats(benz, window, detect_onset(benz, window))
    assert mean == pytest.approx(0.4444, abs=1e-3)
    assert sd == pytest.approx(0.0593, abs=1e-3)


def test_detect_onset_returns_none_when_nothing_deflects() -> None:
    """A flat channel must not produce a spurious onset (cf. opto hexanol)."""
    df = load_pid_session(BENZ_SESSION)
    window = exposure_windows(df)[0]
    onset = detect_onset(df, window, sigma=1e6)
    assert onset.latency_s is None


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def test_figures_render(tmp_path: Path) -> None:
    trace = make_trace_figure(BENZ_SESSION, tmp_path)
    latency = make_latency_figure(tmp_path)
    for paths in (trace, latency):
        for path in paths:
            assert path.exists(), path
            assert path.stat().st_size > 10_000, path
    assert {p.suffix for p in trace} == {".png", ".pdf", ".svg"}
    assert {p.suffix for p in latency} == {".png", ".pdf", ".svg"}
