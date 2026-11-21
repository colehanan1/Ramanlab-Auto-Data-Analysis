"""Pins how previous_odor_carryover pairs each trial with the odor delivered before it."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
import previous_odor_carryover as poc  # noqa: E402


def _preds(rows):
    return pd.DataFrame(rows, columns=["dataset", "fly", "fly_number", "trial_label", "score", "trial_type"])


def _panel(batch, fly, ttype, odors, scores, cohort="IAA-Sensitivity-1-1"):
    prefix = "pretest" if ttype == "pretest" else "testing"
    return [(cohort, batch, fly, f"{prefix}_{i}_{o}", s, ttype)
            for i, (o, s) in enumerate(zip(odors, scores), start=1)]


def test_parse_trials_maps_names_and_threshold():
    t = poc.parse_trials(_preds(_panel("b1", 1, "pretest", ["isoamylacetate", "3-octonol"], [2, 1])))
    assert list(t["odor"]) == ["Isoamyl Acetate", "3-Octanol"]
    assert list(t["block"]) == ["pre", "pre"]
    assert list(t["responded"]) == [True, False]   # score >= 2


def test_previous_odor_is_the_trial_one_position_earlier_same_panel():
    rows = _panel("b1", 1, "pretest", ["ethylbutyrate", "isoamylacetate", "hexanol"], [0, 2, 0])
    tr = poc.build_transitions(poc.parse_trials(_preds(rows)))
    assert list(tr["odor"]) == ["Isoamyl Acetate", "Hexanol"]          # first trial has no previous
    assert list(tr["prev_odor"]) == ["Ethyl Butyrate", "Isoamyl Acetate"]
    assert list(tr["prev_responded"]) == [False, True]


def test_post_trial_1_follows_training_and_light_only_is_never_a_target():
    rows = _panel("b1", 1, "testing", ["hexanol", "citral", "paraffinoil", "lightonly"], [0, 0, 2, 5])
    tr = poc.build_transitions(poc.parse_trials(_preds(rows)))
    # testing_1 follows training (not an odor panel trial) -> dropped; light-only dropped as target
    assert list(tr["odor"]) == ["Citral", "Paraffin Oil"]
    assert list(tr["prev_odor"]) == ["Hexanol", "Citral"]


def test_training_odor_is_the_predecessor_of_post_trial_1_when_given():
    rows = (_panel("b1", 1, "testing", ["isoamylacetate", "citral"], [3, 0])
            + _panel("b2", 1, "testing", ["paraffinoil", "hexanol"], [0, 0],
                     cohort="Paraffin-Sensitivity-1-100"))
    trials = poc.parse_trials(_preds(rows))
    training = pd.DataFrame({"fly_id": ["IAA-Sensitivity-1-1|b1|1"], "last_training_responded": [True]})
    tr = poc.build_transitions(trials, training_odor=poc.SENSITIZED, training_last=training)
    first = tr[tr["pos"] == 1].set_index("cohort")
    # the trained odor sits in front of post trial 1 -- even when it is the same odor
    assert first.loc["IAA-Sensitivity-1-1", "prev_odor"] == "Isoamyl Acetate"
    assert first.loc["IAA-Sensitivity-1-1", "odor"] == "Isoamyl Acetate"
    assert first.loc["Paraffin-Sensitivity-1-100", "prev_odor"] == "Paraffin Oil"
    assert set(first["prev_source"]) == {"training"}
    assert bool(first.loc["IAA-Sensitivity-1-1", "prev_responded"]) is True
    assert pd.isna(first.loc["Paraffin-Sensitivity-1-100", "prev_responded"])
    # panel transitions are unchanged; panel paraffin is still not a predecessor
    later = tr[tr["pos"] == 2]
    assert set(later["prev_source"]) == {"panel"}
    assert list(later["prev_odor"]) == ["Isoamyl Acetate"]


def test_permutation_may_keep_a_training_self_transition_but_never_creates_one_in_a_panel():
    import numpy as np
    rows = []
    for f in (1, 2):
        rows += _panel("b1", f, "testing", ["isoamylacetate", "citral", "hexanol", "linalool"], [0] * 4)
    trials = poc.parse_trials(_preds(rows))
    tr = poc.build_transitions(trials, training_odor=poc.SENSITIZED)
    perm = poc.Permuter(tr, rng=np.random.default_rng(1))
    panel = (tr["prev_source"] == "panel").to_numpy()
    for _ in range(50):
        p = perm.shuffled_prev()
        assert not np.any((p == tr["odor"].to_numpy()) & panel)
        assert sorted(p[tr["fly_number"].to_numpy() == 1]) == sorted(tr.loc[tr["fly_number"] == 1, "prev_odor"])


def test_unparsed_and_unknown_labels_can_be_skipped():
    rows = (_panel("b1", 1, "testing", ["hexanol", "ethylbutyrate"], [0, 2])
            + [("X", "b1", 1, "testing_3_isoamylacetate-1_then_hexanol", 0, "testing")])
    with pytest.raises(ValueError):
        poc.parse_trials(_preds(rows))
    t = poc.parse_trials(_preds(rows), skip_unknown=True)
    assert list(t["odor"]) == ["Hexanol", "Ethyl Butyrate"]


def test_fixed_position_paraffin_is_not_a_randomized_previous_odor():
    rows = _panel("b1", 1, "pretest", ["paraffinoil", "hexanol", "citral"], [0, 0, 0])
    tr = poc.build_transitions(poc.parse_trials(_preds(rows)))
    assert list(tr["prev_odor"]) == ["Hexanol"]                        # hexanol (after paraffin) dropped


def test_previous_odor_comes_from_the_batch_sequence_when_the_fly_lacks_that_trial():
    rows = (_panel("b1", 1, "pretest", ["ethylbutyrate", "isoamylacetate"], [0, 3])
            + [r for r in _panel("b1", 2, "pretest", ["ethylbutyrate", "isoamylacetate"], [1, 0])
               if not r[3].startswith("pretest_1_")])
    tr = poc.build_transitions(poc.parse_trials(_preds(rows)))
    fly2 = tr[tr["fly_number"] == 2]
    assert list(fly2["prev_odor"]) == ["Ethyl Butyrate"]
    assert fly2["prev_responded"].isna().all()                          # unknown, not False


def test_panels_and_flies_do_not_leak_into_each_other():
    rows = (_panel("b1", 1, "pretest", ["hexanol", "citral"], [0, 0])
            + _panel("b1", 1, "testing", ["linalool", "benzaldehyde"], [0, 0])
            + _panel("b2", 1, "pretest", ["linalool", "hexanol"], [0, 0]))
    tr = poc.build_transitions(poc.parse_trials(_preds(rows)))
    got = set(zip(tr["batch"], tr["block"], tr["prev_odor"], tr["odor"]))
    assert got == {("b1", "pre", "Hexanol", "Citral"),
                   ("b1", "post", "Linalool", "Benzaldehyde"),
                   ("b2", "pre", "Linalool", "Hexanol")}


def test_permutation_shuffles_within_each_batch_panel():
    import numpy as np
    seqs = {"b1": ["hexanol", "citral", "linalool", "benzaldehyde", "ethylbutyrate"],
            "b2": ["citral", "isoamylacetate", "hexanol", "3-octonol", "linalool"]}
    rows = []
    for b, seq in seqs.items():
        for f in (1, 2):
            rows += _panel(b, f, "pretest", seq, [0] * len(seq))
    tr = poc.build_transitions(poc.parse_trials(_preds(rows)))
    perm = poc.Permuter(tr, rng=np.random.default_rng(0))
    moved = False
    for _ in range(50):
        p = perm.shuffled_prev()
        # both flies of a batch at one position always carry the same previous odor
        key = tr["batch"] + tr["pos"].astype(str)
        assert pd.Series(p).groupby(key.to_numpy()).nunique().max() == 1
        for b in seqs:
            m = (tr["batch"] == b).to_numpy()
            # each panel keeps its own set of predecessors (each once per fly) ...
            assert sorted(p[m]) == sorted(tr.loc[m, "prev_odor"])
            # ... and no trial is ever given its own odor as predecessor
            assert not np.any(p[m] == tr.loc[m, "odor"].to_numpy())
        moved |= bool(np.any(p != tr["prev_odor"].to_numpy()))
    assert moved


def test_shuffled_codes_match_shuffled_labels():
    import numpy as np
    rows = []
    for f in (1, 2):
        rows += _panel("b1", f, "pretest", ["hexanol", "citral", "linalool", "benzaldehyde"], [0] * 4)
    tr = poc.build_transitions(poc.parse_trials(_preds(rows)))
    a = poc.Permuter(tr, rng=np.random.default_rng(3)).shuffled_prev()
    c = poc.Permuter(tr, rng=np.random.default_rng(3)).shuffled_prev_codes()
    assert list(np.array(poc.RANDOMIZED_ODORS)[c]) == list(a)
