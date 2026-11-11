"""``--window-note`` labels every pre-vs-post figure with the scoring window.

The +5 s set is rendered by the same script from a rescored predictions CSV;
without a note its figures would be indistinguishable from the standard set.
The default (no note) must leave the pipeline's titles untouched.
"""

import pandas as pd
import pytest

import scripts.analysis.pretest_vs_test_comparison as pvt


def _paired():
    rows = []
    for fly in range(8):
        for odor in ("Isoamyl Acetate 1", "Hexanol 0.1"):
            rows.append({"dataset_canon": "IAA-Sensitivity-24-1", "fly": f"b{fly}",
                         "fly_number": "1", "odor": odor,
                         "score_pre": fly % 3, "score_post": (fly + 1) % 5})
    return pd.DataFrame(rows)


@pytest.fixture
def calls(monkeypatch):
    seen = {}

    def record(name):
        def fake(*args, **kwargs):
            seen[name] = kwargs
            return kwargs.get("path")
        return fake

    for name in ("_grouped_bars", "_spotlight", "_slopes", "_heatmap"):
        monkeypatch.setattr(pvt, name, record(name))
    return seen


def test_the_note_reaches_every_figure(calls, tmp_path):
    pvt.render_cohort(_paired(), cohort="IAA-Sensitivity-24-1", out_dir=tmp_path,
                      window_note="scored on odor + 5 s after")
    assert "scored on odor + 5 s after" in calls["_grouped_bars"]["test_note"]
    assert "scored on odor + 5 s after" in calls["_spotlight"]["test_note"]
    assert calls["_slopes"]["note"] == "scored on odor + 5 s after"
    assert calls["_heatmap"]["note"] == "scored on odor + 5 s after"


def test_no_note_by_default(calls, tmp_path):
    pvt.render_cohort(_paired(), cohort="IAA-Sensitivity-24-1", out_dir=tmp_path)
    assert calls["_grouped_bars"]["test_note"] == ""
    assert calls["_slopes"].get("note", "") == ""


def test_the_cli_accepts_the_note():
    args = pvt._parse_args(["--predictions-csv", "p.csv", "--out-root", "o",
                            "--window-note", "scored on odor + 5 s after"])
    assert args.window_note == "scored on odor + 5 s after"
