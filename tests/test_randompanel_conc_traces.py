"""Concentration-series mean traces for datasets with no control arm.

The RandomPanel datasets have no control cohort (``RandomPanel-Control-24-10``
carries zero rows), so the trained-vs-control figure cannot be drawn for them.
What *is* comparable is the same odor at the three delivered concentrations --
10, 1 and 0.1 -- which live in three separate datasets. This driver overlays
those three means on one axes per odor, encoding concentration twice:
darkest-to-lightest shade of the odor's own colour, and solid / dashed / dotted
from high to low, so the ranking survives greyscale printing and colour-vision
deficiency alike.

Both presentations of an odor are pooled per fly (as in
``randompanel_conc_comparison``): three lines, not six.
"""

from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from scripts.analysis import envelope_visuals as ev  # noqa: E402
from scripts.analysis.randompanel_conc_traces import (  # noqa: E402
    conc_series_styles,
    main,
    pool_presentations,
)

DS_10 = "RandomPanel-Training-24-10"
DS_1 = "RandomPanel-24-1"
DS_01 = "RandomPanel-24-0.1"
CONC_BY_DATASET = {DS_10: 10.0, DS_1: 1.0, DS_01: 0.1}

# Each odor twice, as the real RandomPanel schedule presents them.
SCHEDULE = [
    ("testing_1_hexanol", 1.0),
    ("testing_2_ethylbutyrate", 2.0),
    ("testing_3_hexanol", 3.0),
    ("testing_4_ethylbutyrate", 4.0),
]

N_FRAMES = 120
FPS = 40.0
ODOR_ON_S = 0.5
BASELINE_FRAMES = int(round(ODOR_ON_S * FPS))


@pytest.fixture(autouse=True)
def _protocol():
    saved_protocol = ev.get_protocol()
    saved_remap = {ds: dict(m) for ds, m in ev._DATASET_ODOR_REMAP.items()}
    ev.set_protocol("v2")
    ev.set_dataset_odor_remap({})
    try:
        yield
    finally:
        ev.set_protocol(saved_protocol)
        ev.set_dataset_odor_remap(saved_remap)


def _wide_frame(datasets=(DS_10, DS_1, DS_01), schedule=None):
    rows = []
    for dataset in datasets:
        for fly in ("july_20_batch_1", "july_20_batch_2"):
            for fly_number in (1, 2):
                for label, value in (schedule or SCHEDULE):
                    row = {
                        "dataset": dataset,
                        "fly": fly,
                        "fly_number": fly_number,
                        "trial_type": "testing",
                        "trial_label": label,
                    }
                    # Flat baseline then a ramp, so the baseline-corrected mean
                    # is not identically zero (a degenerate shared ylim).
                    row.update(
                        {
                            f"dir_val_{i}": value * max(0, i - BASELINE_FRAMES)
                            for i in range(N_FRAMES)
                        }
                    )
                    rows.append(row)
    return pd.DataFrame(rows)


def _luminance(color) -> float:
    r, g, b = matplotlib.colors.to_rgb(color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _run(tmp_path, frame=None, extra=()):
    wide = tmp_path / "wide.parquet"
    (frame if frame is not None else _wide_frame()).to_parquet(wide)
    out_dir = tmp_path / "figs"
    argv = [
        "--wide-csv", str(wide),
        "--out-dir", str(out_dir),
        "--fps", str(FPS),
        "--odor-on-s", str(ODOR_ON_S),
        "--odor-off-s", "1.0",
    ]
    for dataset, conc in CONC_BY_DATASET.items():
        argv.extend(["--dataset", f"{dataset}={conc:g}"])
    argv.extend(extra)
    main(argv)
    return out_dir


# ---------------------------------------------------------------------------
# Styling: shade and dash both encode concentration
# ---------------------------------------------------------------------------


def test_high_concentration_is_solid_and_low_is_dotted():
    styles = conc_series_styles([0.1, 1.0, 10.0], "#1f77b4")
    assert styles[10.0].linestyle == "-"
    assert styles[1.0].linestyle == "--"
    assert styles[0.1].linestyle == ":"


def test_shade_darkens_with_concentration():
    styles = conc_series_styles([0.1, 1.0, 10.0], "#1f77b4")
    lums = [_luminance(styles[c].color) for c in (10.0, 1.0, 0.1)]
    assert lums[0] < lums[1] < lums[2], lums


def test_styles_are_ordered_high_to_low_whatever_the_input_order():
    """Draw order decides which line sits on top; the darkest must not be
    buried under the lightest."""
    assert list(conc_series_styles([0.1, 10.0, 1.0], "#1f77b4")) == [10.0, 1.0, 0.1]


def test_two_concentrations_still_get_the_extreme_shades():
    styles = conc_series_styles([1.0, 10.0], "#1f77b4")
    assert styles[10.0].linestyle == "-"
    assert styles[1.0].linestyle == "--"
    assert _luminance(styles[10.0].color) < _luminance(styles[1.0].color)


def test_more_series_than_linestyles_does_not_crash():
    styles = conc_series_styles([0.01, 0.1, 1.0, 10.0], "#1f77b4")
    assert len(styles) == 4
    assert styles[10.0].linestyle == "-"


# ---------------------------------------------------------------------------
# Pooling the two presentations
# ---------------------------------------------------------------------------


def test_presentations_are_pooled_per_fly():
    per_key = {
        "Hexanol 1": {"fly_a": np.array([1.0, 1.0]), "fly_b": np.array([2.0, 2.0])},
        "Hexanol 2": {"fly_a": np.array([3.0, 3.0])},
    }
    pooled = pool_presentations(per_key)
    assert set(pooled) == {"Hexanol"}
    assert pooled["Hexanol"]["fly_a"].tolist() == [2.0, 2.0]   # (1 + 3) / 2
    assert pooled["Hexanol"]["fly_b"].tolist() == [2.0, 2.0]   # only one exposure


def test_unnumbered_odors_pass_through():
    pooled = pool_presentations({"Citral": {"fly_a": np.array([1.0])}})
    assert pooled["Citral"]["fly_a"].tolist() == [1.0]


def test_pooling_tolerates_ragged_trace_lengths():
    """A short trial must not truncate the fly's other exposure to nothing."""
    per_key = {
        "Hexanol 1": {"fly_a": np.array([2.0, 2.0, 2.0])},
        "Hexanol 2": {"fly_a": np.array([4.0])},
    }
    pooled = pool_presentations(per_key)
    assert pooled["Hexanol"]["fly_a"][0] == 3.0
    assert len(pooled["Hexanol"]["fly_a"]) == 3


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


def test_one_figure_per_odor_plus_a_sidecar(tmp_path):
    out_dir = _run(tmp_path)
    names = sorted(p.name for p in out_dir.iterdir())
    assert names == [
        "Ethyl_Butyrate_conc_series.png",
        "Hexanol_conc_series.png",
        "conc_series.json",
    ]


def test_sidecar_records_n_per_concentration(tmp_path):
    out_dir = _run(tmp_path)
    meta = json.loads((out_dir / "conc_series.json").read_text())
    assert meta["datasets"] == {k: v for k, v in CONC_BY_DATASET.items()}
    hexanol = meta["per_odor"]["Hexanol"]
    assert hexanol == {"10": 4, "1": 4, "0.1": 4}


def test_an_odor_in_only_one_dataset_is_skipped_not_drawn_alone(tmp_path):
    """A single line is not a concentration comparison; it would read as one."""
    frame = pd.concat(
        [
            _wide_frame(datasets=(DS_10,)),
            _wide_frame(
                datasets=(DS_1, DS_01),
                schedule=[("testing_1_hexanol", 1.0), ("testing_2_hexanol", 2.0)],
            ),
        ],
        ignore_index=True,
    )
    out_dir = _run(tmp_path, frame=frame)
    meta = json.loads((out_dir / "conc_series.json").read_text())
    assert meta["skipped_odors"] == ["Ethyl Butyrate"]
    assert not (out_dir / "Ethyl_Butyrate_conc_series.png").exists()


def test_every_concentration_present_is_drawn(tmp_path, monkeypatch):
    drawn: list[dict] = []
    from scripts.analysis import randompanel_conc_traces as mod

    real = mod.plot_conc_series_for_odor

    def _spy(**kwargs):
        drawn.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(mod, "plot_conc_series_for_odor", _spy)
    _run(tmp_path)
    assert drawn, "no figure was drawn"
    for call in drawn:
        assert [s.conc for s in call["series"]] == [10.0, 1.0, 0.1]
        assert [s.linestyle for s in call["series"]] == ["-", "--", ":"]
    plt.close("all")


def test_the_odor_window_is_shaded_and_the_legend_names_the_concentrations(tmp_path):
    from scripts.analysis.randompanel_conc_traces import (
        ConcSeries,
        plot_conc_series_for_odor,
    )

    series = [
        ConcSeries(
            conc=c,
            per_fly={"fly_a": np.full(80, float(i))},
            color="#08306b",
            linestyle=ls,
        )
        for i, (c, ls) in enumerate(((10.0, "-"), (1.0, "--"), (0.1, ":")))
    ]
    fig = plot_conc_series_for_odor(
        odor="Hexanol",
        series=series,
        fps=FPS,
        odor_on_s=0.5,
        odor_off_s=1.0,
        ylim=(-10.0, 10.0),
    )
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["10% (n=1)", "1% (n=1)", "0.1% (n=1)"]
    assert "Hexanol" in ax.get_title()
    plt.close(fig)


def test_shared_ylim_across_odors(tmp_path):
    """Two figures on different scales cannot be compared by eye."""
    out_dir = _run(tmp_path)
    meta = json.loads((out_dir / "conc_series.json").read_text())
    assert meta["shared_mean_ylim"][0] < meta["shared_mean_ylim"][1]


def test_a_missing_dataset_is_reported_not_silently_dropped(tmp_path):
    with pytest.raises(SystemExit, match="RandomPanel-24-1"):
        _run(
            tmp_path,
            frame=_wide_frame(datasets=(DS_10, DS_01)),
        )
