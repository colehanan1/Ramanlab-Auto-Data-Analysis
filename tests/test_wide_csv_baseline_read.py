"""build_wide_csv reads each baseline (testing) table once, not twice.

The baseline pass read the first odor-on frames of every testing table
(``nrows``) and the main loop then read the whole table again. Now the
baseline pass reads the table once, with the main loop's columns, takes its
sample from those rows and hands the frame on to the main loop -- but only
when the measure column came back numeric. Then the first ``nrows`` values
are exactly those of the ``nrows``-limited read: the C parser converts each
token on its own, whatever follows it. Anything else (a stray token turning
the column object, say) takes the old ``nrows``-limited read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.analysis.envelope_combined as ec
import scripts.analysis.envelope_visuals as ev

MEASURE = "envelope_of_rms"


def _floats(series: pd.Series) -> np.ndarray:
    """What build_wide_csv does to a measure column before using it."""
    return pd.to_numeric(series, errors="coerce").astype(float).to_numpy()


@pytest.mark.parametrize(
    "body, reusable",
    [
        ("0.1\n0.2\n0.30000000000000004\n1e-05\n7\n", True),
        ("1\n2\n3\n4.5\n", True),  # the nrows read infers int64, the full one float64
        ("\n\n\n1.5\n", True),  # all-empty window
        ("0.1\ninf\nnan\n-inf\n2\n", True),
        ("1.23456789012345678901234\n-0.0\n5e-324\n7\n", True),
        ("1\n2\n3\n18446744073709551615\n", True),  # full read uint64
        ("0.1\n0.2\n0.3\n1e400\n", None),  # pandas 2.3: object; 3.0: inf
        ("0.1\n0.2\n0.3\nabc\n", False),  # stray token after the window: object
        ("0.1\nabc\n0.3\n0.4\n", False),
        ("True\nFalse\nTrue\nFalse\n", False),
    ],
)
def test_baseline_rows_equal_an_nrows_limited_read(tmp_path, body, reusable):
    """``reusable`` None: version-dependent -- either way is fine, but rows
    that are handed back must be the nrows-limited read's."""
    p = tmp_path / "t.csv"
    p.write_text(f"time_s,{MEASURE}\n" + "".join(f"{i},{v}\n" for i, v in enumerate(body.splitlines())))
    full = ec._read_trial_table(p, columns=[MEASURE])
    got = ec._baseline_rows(full, MEASURE, 3)
    if reusable is False or (reusable is None and got is None):
        assert got is None
        return
    assert got is not None
    want = ec._read_trial_table(p, columns=[MEASURE], nrows=3)
    assert _floats(got[MEASURE]).tobytes() == _floats(want[MEASURE]).tobytes()


def test_baseline_rows_parquet_and_missing_column(tmp_path):
    p = tmp_path / "t.parquet"
    pd.DataFrame({MEASURE: [0.5, np.nan, 2.25, 9.0]}).to_parquet(p, index=False)
    full = ec._read_trial_table(p, columns=[MEASURE])
    got = ec._baseline_rows(full, MEASURE, 2)
    want = ec._read_trial_table(p, columns=[MEASURE], nrows=2)
    assert got.equals(want)
    assert ec._baseline_rows(full, "missing", 2) is None


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def _write(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({MEASURE: values}).to_csv(path, index=False)


def _make_live(root: Path) -> Path:
    rng = np.random.default_rng(11)
    for batch in ("october_01_batch_1_rig_2", "october_02_batch_1"):
        env = root / batch / "angle_distance_rms_envelope"
        for tt, idx in (("pretest", 1), ("training", 1), ("testing", 1), ("testing", 2)):
            for fly in (1, 2):
                vals = rng.normal(30.0, 10.0, 1400 + 3 * idx)  # longer than the 1260/1200-frame windows
                vals[::13] = np.nan
                stem = f"{tt}_{idx}_hexanol_fly{fly}_distances_fly{fly}_angle_distance_rms_envelope"
                _write(env / f"{stem}.csv", vals)
    # A testing table whose measure column turns object past the baseline
    # window: its sample must still come from the nrows-limited read.
    junk = root / "october_02_batch_1" / "angle_distance_rms_envelope"
    path = junk / "testing_3_hexanol_fly1_distances_fly1_angle_distance_rms_envelope.csv"
    cells = [repr(float(v)) for v in rng.normal(30.0, 10.0, 1400)]
    cells[1350] = "1.5.3"
    path.write_text(f"{MEASURE}\n" + "\n".join(cells) + "\n")
    return root


def _build(tmp_path: Path, tag: str) -> dict[str, bytes]:
    out = tmp_path / tag
    out.mkdir()
    ec.build_wide_csv(
        [str(tmp_path / "LIVE-DS")],
        str(out / "w.csv"),
        measure_cols=[MEASURE],
        extra_trial_exports={"training": str(out / "t.csv"), "pretest": str(out / "p.csv")},
    )
    return {p.name: p.read_bytes() for p in sorted(out.iterdir())}


@pytest.mark.parametrize("protocol", ["v2", "legacy"])
def test_each_baseline_table_is_read_once_and_the_bytes_do_not_change(tmp_path, monkeypatch, protocol):
    ev.set_protocol(protocol)
    _make_live(tmp_path / "LIVE-DS")
    reads: list[tuple[str, object]] = []
    real = ec._read_trial_table

    def _spy(path, **kw):
        reads.append((str(path), kw.get("nrows")))
        return real(path, **kw)

    monkeypatch.setattr(ec, "_read_trial_table", _spy)
    once = _build(tmp_path, "once")
    once_reads = list(reads)

    # Reference: the pre-change flow -- an nrows-limited baseline read, then
    # the main loop's full read.
    monkeypatch.setattr(ec, "_baseline_rows", lambda *a, **k: None)
    reads.clear()
    twice = _build(tmp_path, "twice")

    assert once == twice
    per_table = {}
    for name, nrows in once_reads:
        per_table.setdefault(name, []).append(nrows)
    for name, calls in per_table.items():
        if Path(name).name.startswith("testing_3"):
            # full read (object) -> nrows read -> the main loop reads again
            assert calls[0] is None and calls[1] is not None and calls[2:] == [None], calls
        else:
            assert calls == [None], (name, calls)
    assert len(per_table) == 2 * 4 * 2 + 1
    assert sum(1 for _, nrows in reads if nrows is not None) == 2 * 2 * 2 + 1  # every testing table
