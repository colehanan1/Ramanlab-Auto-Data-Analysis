"""Per-trial light-stimulus check annotations on the Raw PER trace figures.

The ``check_light_stimulus`` pipeline step verifies (from the trial video) that
the LED physically stayed on for its sensor-commanded window and records
``fraction_on_in_window`` per trial in ``logs/light_stimulus_flags.csv``. These
tests pin the plumbing that surfaces that verdict on the Raw-Testing /
Raw-Training PER trace figures:

  * ``load_light_check_fractions`` reads the QC CSV into a mapping keyed by
    (dataset, batch dir name, phase, trial number) — the batch dir name equals
    the wide table's ``fly`` column, and phase/number are parsed from
    ``trial_label``;
  * ``set_light_check_fractions`` registers the canonical dataset spelling as
    an alias (mirrors ``set_model_scores`` — "3Oct-…" vs "3OCT-…");
  * ``_light_check_annotation`` renders ">= 95% of the window" as a green
    checkmark "full" note and anything lower as a red percentage note;
  * ``generate_envelope_plots`` draws the annotation on trials with a
    registered check and never on control-dataset panels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.axes
import pandas as pd
import pytest

from fbpipe.odor_constants import canon_dataset
from scripts.analysis import envelope_visuals as ev

RAW_DATASET = "3Oct-Training-24-0.1"
CANON_DATASET = "3OCT-Training-24-0.1"
FLY = "july_20_batch_1_rig_2"


@pytest.fixture()
def restore_light_checks():
    saved = dict(ev._LIGHT_CHECKS)
    try:
        yield
    finally:
        ev.set_light_check_fractions(saved)


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def _write_flags_csv(path: Path, rows: list[dict]) -> None:
    defaults = {
        "dataset": RAW_DATASET,
        "batch": f"/data/{RAW_DATASET}/{FLY}",
        "trial_type": "training",
        "trial_index": 1,
        "odor": "3-Octanol",
        "trial_dir": "unused",
        "status": "checked",
        "fraction_on_in_window": 1.0,
        "passed": True,
    }
    pd.DataFrame([{**defaults, **row} for row in rows]).to_csv(path, index=False)


def test_loader_keys_by_dataset_batchname_phase_and_number(tmp_path):
    csv_path = tmp_path / "flags.csv"
    _write_flags_csv(
        csv_path,
        [
            {"trial_index": 1, "fraction_on_in_window": 1.0},
            {"trial_index": 2, "fraction_on_in_window": 0.62, "trial_type": "testing"},
        ],
    )
    fractions = ev.load_light_check_fractions(csv_path)
    assert fractions == {
        (RAW_DATASET, FLY, "training", 1): 1.0,
        (RAW_DATASET, FLY, "testing", 2): 0.62,
    }


def test_loader_uses_batch_basename_not_full_path(tmp_path):
    """The wide table's ``fly`` column is the batch dir NAME; keys must match it."""
    csv_path = tmp_path / "flags.csv"
    _write_flags_csv(csv_path, [{"batch": "/very/deep/path/july_20_batch_1_rig_2"}])
    keys = list(ev.load_light_check_fractions(csv_path))
    assert keys == [(RAW_DATASET, "july_20_batch_1_rig_2", "training", 1)]


def test_loader_skips_unchecked_and_unparseable_rows(tmp_path):
    csv_path = tmp_path / "flags.csv"
    _write_flags_csv(
        csv_path,
        [
            {"status": "no_video", "fraction_on_in_window": float("nan")},
            {"status": "unreadable_video", "fraction_on_in_window": float("nan")},
            {"trial_index": 3, "fraction_on_in_window": float("nan")},
            {"trial_index": 4, "fraction_on_in_window": 0.5},
        ],
    )
    fractions = ev.load_light_check_fractions(csv_path)
    assert fractions == {(RAW_DATASET, FLY, "training", 4): 0.5}


def test_loader_missing_csv_returns_empty(tmp_path):
    assert ev.load_light_check_fractions(tmp_path / "nope.csv") == {}


# ---------------------------------------------------------------------------
# Registry + lookup
# ---------------------------------------------------------------------------

def test_lookup_resolves_canonical_dataset_alias(restore_light_checks):
    assert canon_dataset(RAW_DATASET) == CANON_DATASET != RAW_DATASET
    ev.set_light_check_fractions({(RAW_DATASET, FLY, "training", 3): 0.97})
    assert ev._lookup_light_check(CANON_DATASET, FLY, "training_3") == 0.97
    assert ev._lookup_light_check(RAW_DATASET, FLY, "training_3") == 0.97


def test_lookup_parses_phase_and_number_from_trial_label(restore_light_checks):
    ev.set_light_check_fractions(
        {
            (RAW_DATASET, FLY, "testing", 9): 0.4,
            (RAW_DATASET, FLY, "training", 12): 1.0,
        }
    )
    # Odor-suffixed and bare labels both resolve to the rig-side phase/number.
    assert ev._lookup_light_check(CANON_DATASET, FLY, "testing_9_lightonly") == 0.4
    assert ev._lookup_light_check(CANON_DATASET, FLY, "training_12") == 1.0
    # Labels without a leading phase token can't join to the QC CSV.
    assert ev._lookup_light_check(CANON_DATASET, FLY, "odd_label") is None


def test_lookup_misses_other_fly_and_trial(restore_light_checks):
    ev.set_light_check_fractions({(RAW_DATASET, FLY, "training", 3): 0.97})
    assert ev._lookup_light_check(CANON_DATASET, "june_01_batch_9", "training_3") is None
    assert ev._lookup_light_check(CANON_DATASET, FLY, "training_4") is None


def test_zero_fraction_survives_registration(restore_light_checks):
    """0.0 (light never came on) is a real result, not a missing one."""
    ev.set_light_check_fractions({(RAW_DATASET, FLY, "training", 3): 0.0})
    assert ev._lookup_light_check(CANON_DATASET, FLY, "training_3") == 0.0


# ---------------------------------------------------------------------------
# Annotation formatting (the 95% "full" rule)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fraction", [1.0, 0.999, 0.95])
def test_annotation_full_at_or_above_95(fraction):
    text, color = ev._light_check_annotation(fraction)
    assert "full" in text
    assert "checkmark" in text  # mathtext glyph — Arial has no U+2713
    assert color == ev.LIGHT_CHECK_PASS_COLOR


@pytest.mark.parametrize(
    ("fraction", "pct"), [(0.9499, "94%"), (0.62, "62%"), (0.0, "0%")]
)
def test_annotation_notes_percentage_below_95(fraction, pct):
    text, color = ev._light_check_annotation(fraction)
    assert pct in text
    assert "full" not in text
    assert color == ev.LIGHT_CHECK_WARN_COLOR


def test_annotation_never_rounds_a_failing_fraction_up_to_full():
    """0.9499 must not display as 95% — that would contradict the warn color."""
    text, _ = ev._light_check_annotation(0.9499)
    assert "95" not in text


# ---------------------------------------------------------------------------
# Figure integration
# ---------------------------------------------------------------------------

def _write_wide_csv(path: Path, *, dataset: str, trial_type: str = "training") -> None:
    rows = []
    for trial_num in (1, 2, 3):
        rows.append(
            {
                "dataset": dataset,
                "fly": FLY,
                "fly_number": "1",
                "fly_type": "GR5a-Old",
                "trial_type": trial_type,
                "trial_label": f"{trial_type}_{trial_num}",
                "fps": 40.0,
                "global_min": 1.0,
                "global_max": 25.0,
                "trimmed_global_min": 1.0,
                "trimmed_global_max": 25.0,
                "trace_len": 4,
                "dir_val_0": 0.0 + trial_num,
                "dir_val_1": 8.5,
                "dir_val_2": 16.25,
                "dir_val_3": 4.0,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _render_cfg(wide_input: Path, out_dir: Path, *, trial_type: str) -> ev.EnvelopePlotConfig:
    return ev.EnvelopePlotConfig(
        matrix_npy=Path("/does/not/exist/combined_base.npy"),
        codes_json=Path("/does/not/exist/code_maps.json"),
        out_dir=out_dir,
        latency_sec=0.0,
        odor_latency_s=0.0,
        trial_type=trial_type,
        wide_input=wide_input,
        overwrite=True,
    )


@pytest.fixture()
def spy_axes_text(monkeypatch):
    calls: list[str] = []
    original = matplotlib.axes.Axes.text

    def recording_text(self, *args, **kwargs):
        if len(args) >= 3:
            calls.append(str(args[2]))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "text", recording_text)
    return calls


def test_light_annotations_drawn_on_training_figure(
    tmp_path, restore_light_checks, spy_axes_text
):
    ev.set_protocol("v2")
    wide_csv = tmp_path / "wide.csv"
    _write_wide_csv(wide_csv, dataset="Hex-Training-24-0.01")
    ev.set_light_check_fractions(
        {
            ("Hex-Training-24-0.01", FLY, "training", 1): 1.0,
            ("Hex-Training-24-0.01", FLY, "training", 2): 0.62,
            # training_3 deliberately unregistered -> no annotation.
        }
    )

    ev.generate_envelope_plots(
        _render_cfg(wide_csv, tmp_path / "plots", trial_type="training")
    )

    light_texts = [t for t in spy_axes_text if "Light" in t]
    assert len(light_texts) == 2, light_texts
    assert any("full" in t for t in light_texts)
    assert any("62%" in t for t in light_texts)


def test_no_light_annotation_on_control_datasets(
    tmp_path, restore_light_checks, spy_axes_text
):
    """Control cohorts suppress the light line; the check note must follow."""
    ev.set_protocol("v2")
    wide_csv = tmp_path / "wide.csv"
    _write_wide_csv(wide_csv, dataset="Hex-Control-24-0.01")
    ev.set_light_check_fractions(
        {("Hex-Control-24-0.01", FLY, "training", 1): 1.0}
    )

    ev.generate_envelope_plots(
        _render_cfg(wide_csv, tmp_path / "plots", trial_type="training")
    )

    assert not [t for t in spy_axes_text if "Light" in t]


def test_no_annotation_when_registry_empty(tmp_path, restore_light_checks, spy_axes_text):
    ev.set_protocol("v2")
    ev.set_light_check_fractions({})
    wide_csv = tmp_path / "wide.csv"
    _write_wide_csv(wide_csv, dataset="Hex-Training-24-0.01")

    ev.generate_envelope_plots(
        _render_cfg(wide_csv, tmp_path / "plots", trial_type="training")
    )

    assert not [t for t in spy_axes_text if "Light" in t]


# ---------------------------------------------------------------------------
# Light-Only trace figures (light_trial_traces.py) share the registry
# ---------------------------------------------------------------------------

def _write_light_only_csv(path: Path, *, dataset: str, fly: str) -> None:
    import numpy as np

    n_cols = 1300  # ~32 s @ 40 fps, covers light on @ 30 s
    rows = []
    for fly_number in (1, 2):
        for trial_label, light_on in (
            ("training_15", 30.0),  # Solid 1 Hz (100% duty)
            ("training_16", 30.0),  # Pulse 5 Hz (50% duty)
            ("training_20", 30.0),  # no condition mapping
            ("training_1_acv", None),
        ):
            row = {
                "dataset": dataset,
                "fly": fly,
                "fly_number": fly_number,
                "trial_label": trial_label,
                "trial_light_on_s": light_on if light_on is not None else np.nan,
                "trace_len": n_cols,
                "fps": 40.0,
            }
            for i in range(n_cols):
                row[f"dir_val_{i}"] = 5.0
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_light_only_figures_annotate_from_shared_registry(
    tmp_path, restore_light_checks, spy_axes_text
):
    from scripts.analysis import light_trial_traces as lt

    _write_light_only_csv(
        tmp_path / "wide.csv", dataset="RandomPanel-24-0.1", fly="sess_1"
    )
    ev.set_light_check_fractions(
        {
            ("RandomPanel-24-0.1", "sess_1", "training", 15): 1.0,
            ("RandomPanel-24-0.1", "sess_1", "training", 20): 0.4,
        }
    )

    lt.generate(tmp_path / "wide.csv", tmp_path / "out")

    light_texts = [t for t in spy_axes_text if "Light:" in t]
    assert any("full" in t for t in light_texts), light_texts
    assert any("40%" in t for t in light_texts), light_texts


def test_light_only_pulse_trials_never_annotated(
    tmp_path, restore_light_checks, spy_axes_text
):
    """Trials 16-19 command PULSED light; the stride-sampled video QC cannot
    measure a pulsing LED and the "% of window" rule doesn't apply, so even a
    registered fraction must not draw (a red "0%" there reads as a hardware
    failure on a light that demonstrably pulsed)."""
    from scripts.analysis import light_trial_traces as lt

    _write_light_only_csv(
        tmp_path / "wide.csv", dataset="RandomPanel-24-0.1", fly="sess_1"
    )
    ev.set_light_check_fractions(
        {("RandomPanel-24-0.1", "sess_1", "training", 16): 0.5}
    )

    lt.generate(tmp_path / "wide.csv", tmp_path / "out")

    assert not [t for t in spy_axes_text if "Light:" in t]


def test_light_only_figures_skip_unregistered_trials(
    tmp_path, restore_light_checks, spy_axes_text
):
    from scripts.analysis import light_trial_traces as lt

    _write_light_only_csv(
        tmp_path / "wide.csv", dataset="RandomPanel-24-0.1", fly="sess_1"
    )
    ev.set_light_check_fractions({})

    lt.generate(tmp_path / "wide.csv", tmp_path / "out")

    assert not [t for t in spy_axes_text if "Light:" in t]
