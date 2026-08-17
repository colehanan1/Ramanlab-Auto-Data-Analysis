"""Per-rig/date proboscis gate overrides.

Flybehavior2's camera zoom puts genuine full proboscis extension well past the
global 160 px gates — a verified fly (august_11_batch_4 training_2) reaches
251.3 px, and the 160 px gate silently blanked 832 real frames of that trial.
The fix is a ``rig_gate_overrides:`` config list: recordings from a named host
AFTER a given date get wider gates, resolved per batch directory from the
``Host:`` line in ``session_metadata.txt`` plus the sidecar filename dates.

Semantics pinned here:

* host match is case-insensitive (metadata says ``Flybehavior2``),
* ``after`` is strict — a batch recorded ON the cutoff date keeps defaults,
* batches with no metadata / unknown host / no date keep defaults (safe),
* the resolver returns a modified COPY of Settings; the original is untouched.
"""

from __future__ import annotations

import textwrap
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fbpipe.config import load_settings
from fbpipe.utils.rig_gates import (
    RigGateOverride,
    apply_rig_gate_overrides,
    read_batch_date,
    read_batch_host,
)

SESSION_METADATA = textwrap.dedent(
    """\
    Session Metadata
    ===============

    Run Context
    -----------
    Intake Logged (UTC): 2026-08-11T20:02:09.137828Z
    Host: Flybehavior2 | OS: Linux 6.12.34+rpt-rpi-2712
    """
)


def make_batch(tmp_path: Path, *, host_line: str | None = "Host: Flybehavior2 | OS: Linux",
               sidecar_stamp: str | None = "20260811_163740",
               name: str = "august_11_batch_4") -> Path:
    batch = tmp_path / name
    batch.mkdir(parents=True, exist_ok=True)
    if host_line is not None:
        meta = SESSION_METADATA.replace("Host: Flybehavior2 | OS: Linux 6.12.34+rpt-rpi-2712", host_line)
        (batch / "session_metadata.txt").write_text(meta, encoding="utf-8")
    if sidecar_stamp is not None:
        (batch / f"output_august_11_batch_4_training_2_3-Octonol_{sidecar_stamp}.csv").write_text(
            "frame\n0\n", encoding="utf-8"
        )
    return batch


def make_settings(tmp_path: Path, overrides_yaml: str = "") -> object:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            f"""\
            model_path: /dev/null
            main_directories:
            - {tmp_path}
            distance_limits:
              class2_min: 10.0
              class2_max: 160.0
              three_fly_max_eye_prob_distance_px: 160.0
            proboscis_filter:
              enabled: true
              max_eye_prob_distance_px: 160.0
              up_divisor: 4.0
              max_jump_px: 80.0
            """
        )
        + textwrap.dedent(overrides_yaml),
        encoding="utf-8",
    )
    return load_settings(cfg_path)


FLYBEHAVIOR2_270 = """\
rig_gate_overrides:
- host: Flybehavior2
  after: 2026-07-25
  max_eye_prob_distance_px: 270.0
  class2_max: 270.0
  three_fly_max_eye_prob_distance_px: 270.0
"""


# ── metadata parsing ─────────────────────────────────────────────────


def test_read_batch_host_parses_metadata(tmp_path):
    batch = make_batch(tmp_path)
    assert read_batch_host(batch) == "Flybehavior2"


def test_read_batch_host_missing_metadata(tmp_path):
    batch = make_batch(tmp_path, host_line=None)
    assert read_batch_host(batch) is None


def test_read_batch_date_from_sidecar_filenames(tmp_path):
    batch = make_batch(tmp_path)
    assert read_batch_date(batch) == date(2026, 8, 11)


def test_read_batch_date_uses_earliest_sidecar(tmp_path):
    batch = make_batch(tmp_path)
    (batch / "output_other_trial_20260810_090000.csv").write_text("frame\n", encoding="utf-8")
    assert read_batch_date(batch) == date(2026, 8, 10)


def test_read_batch_date_falls_back_to_metadata(tmp_path):
    batch = make_batch(tmp_path, sidecar_stamp=None)
    assert read_batch_date(batch) == date(2026, 8, 11)


def test_read_batch_date_none_when_unknown(tmp_path):
    batch = make_batch(tmp_path, host_line=None, sidecar_stamp=None)
    assert read_batch_date(batch) is None


# ── config parsing ───────────────────────────────────────────────────


def test_config_parses_rig_gate_overrides(tmp_path):
    cfg = make_settings(tmp_path, FLYBEHAVIOR2_270)
    assert len(cfg.rig_gate_overrides) == 1
    ov = cfg.rig_gate_overrides[0]
    assert isinstance(ov, RigGateOverride)
    assert ov.host == "Flybehavior2"
    assert ov.after == date(2026, 7, 25)
    assert ov.max_eye_prob_distance_px == 270.0
    assert ov.class2_max == 270.0
    assert ov.three_fly_max_eye_prob_distance_px == 270.0


def test_config_without_overrides_is_empty(tmp_path):
    cfg = make_settings(tmp_path)
    assert cfg.rig_gate_overrides == ()


# ── resolution ───────────────────────────────────────────────────────


def test_override_applies_to_matching_batch(tmp_path):
    cfg = make_settings(tmp_path, FLYBEHAVIOR2_270)
    batch = make_batch(tmp_path)
    eff = apply_rig_gate_overrides(cfg, batch)
    assert eff.proboscis_filter.max_eye_prob_distance_px == 270.0
    assert eff.class2_max == 270.0
    assert eff.three_fly_max_eye_prob_distance_px == 270.0
    # original untouched
    assert cfg.proboscis_filter.max_eye_prob_distance_px == 160.0
    assert cfg.class2_max == 160.0


def test_override_host_match_is_case_insensitive(tmp_path):
    cfg = make_settings(tmp_path, FLYBEHAVIOR2_270.replace("Flybehavior2", "FLYBEHAVIOR2"))
    batch = make_batch(tmp_path)
    assert apply_rig_gate_overrides(cfg, batch).class2_max == 270.0


def test_override_ignores_other_host(tmp_path):
    cfg = make_settings(tmp_path, FLYBEHAVIOR2_270)
    batch = make_batch(tmp_path, host_line="Host: BehaviorLocust | OS: Linux")
    eff = apply_rig_gate_overrides(cfg, batch)
    assert eff.proboscis_filter.max_eye_prob_distance_px == 160.0
    assert eff.class2_max == 160.0


def test_override_is_strictly_after_cutoff(tmp_path):
    cfg = make_settings(tmp_path, FLYBEHAVIOR2_270)
    on_cutoff = make_batch(tmp_path, sidecar_stamp="20260725_120000", name="july_25_batch_1")
    assert apply_rig_gate_overrides(cfg, on_cutoff).class2_max == 160.0
    before = make_batch(tmp_path, sidecar_stamp="20260720_120000", name="july_20_batch_1")
    assert apply_rig_gate_overrides(cfg, before).class2_max == 160.0
    after = make_batch(tmp_path, sidecar_stamp="20260726_000100", name="july_26_batch_1")
    assert apply_rig_gate_overrides(cfg, after).class2_max == 270.0


def test_override_skips_batch_without_metadata(tmp_path):
    cfg = make_settings(tmp_path, FLYBEHAVIOR2_270)
    batch = make_batch(tmp_path, host_line=None)
    eff = apply_rig_gate_overrides(cfg, batch)
    assert eff.class2_max == 160.0


def test_no_overrides_returns_cfg_unchanged(tmp_path):
    cfg = make_settings(tmp_path)
    batch = make_batch(tmp_path)
    assert apply_rig_gate_overrides(cfg, batch) is cfg


# ── step wiring: reject_bad_proboscis honors the per-batch gate ──────


def _write_distances_parquet(batch: Path) -> Path:
    trial = batch / "august_11_batch_4_training_2"
    trial.mkdir(exist_ok=True)
    # Eye fixed at (500, 400); proboscis straight DOWN (generous axis) at
    # 100 → 175 → 250 px, stepping ≤ max_jump_px so only the GEOMETRY gate is
    # in play. 250 px is a REAL extension on this rig: outside the 160 gate,
    # inside 270.
    df = pd.DataFrame(
        {
            "frame": [0, 1, 2],
            "x_class0": [500.0, 500.0, 500.0],
            "y_class0": [400.0, 400.0, 400.0],
            "track_id_class1": [11.0, 11.0, 11.0],
            "x_class1": [500.0, 500.0, 500.0],
            "y_class1": [500.0, 575.0, 650.0],
            "distance_0_1": [100.0, 175.0, 250.0],
        }
    )
    path = trial / "august_11_batch_4_training_2_fly1_distances.parquet"
    df.to_parquet(path)
    return path


@pytest.mark.parametrize(
    "overrides_yaml, expect_250_kept",
    [(FLYBEHAVIOR2_270, True), ("", False)],
    ids=["with-270-override", "default-160-gate"],
)
def test_reject_step_uses_per_batch_gate(tmp_path, overrides_yaml, expect_250_kept):
    from fbpipe.steps.reject_bad_proboscis import _process_fly_dir

    cfg = make_settings(tmp_path, overrides_yaml)
    batch = make_batch(tmp_path)
    parquet = _write_distances_parquet(batch)

    _process_fly_dir(batch, cfg)

    out = pd.read_parquet(parquet)
    d = pd.to_numeric(out["distance_0_1"], errors="coerce")
    assert d.iloc[0] == 100.0  # inside every gate, always kept
    if expect_250_kept:
        assert d.iloc[1] == 175.0
        assert d.iloc[2] == 250.0
    else:
        assert np.isnan(d.iloc[1])
        assert np.isnan(d.iloc[2])


def test_distance_stats_uses_per_batch_class2_max(tmp_path):
    from fbpipe.steps.distance_stats import _process_fly_dir

    cfg = make_settings(tmp_path, FLYBEHAVIOR2_270)
    batch = make_batch(tmp_path)
    _write_distances_parquet(batch)

    _process_fly_dir(batch, cfg)

    import json

    stats_path = batch / "fly1_global_distance_stats_class_0.json"
    assert stats_path.exists()
    stats = json.loads(stats_path.read_text())
    # With the 270 override the 250 px point is inside [class2_min, class2_max]
    # and must set the fly max; under the old 160 cap it was masked out.
    assert stats["global_max"] == 250.0
