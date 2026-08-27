"""An unsorted row order, for the trained arms.

The control rasters rank flies by conditioning vigor, which is the question
those figures ask: did the flies that extended most during odor-only exposure
also react at test?

For a TRAINED cohort that ranking is not neutral. The odor was paired with
light, so PER during conditioning is partly the light-evoked response rather
than a trait of the fly, and ordering the testing rows by it manufactures a
gradient the data need not contain. ``--sort-by random`` drops the ranking.

Random, but not irreproducible: the shuffle is seeded from the dataset name, so
a rerun redraws the same figure and the training and testing panels of one
cohort keep the same rows in the same places. The seed is recorded alongside the
threshold in the JSON sidecar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src"), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from scripts.analysis import binarized_per_rasters as mod  # noqa: E402
from test_binarized_per_rasters import DATASET, _training_frame  # noqa: E402


@pytest.fixture()
def trials():
    return mod.build_trials(_training_frame(), dataset=DATASET, trial_type="training")


# ── the sort key exists ───────────────────────────────────────────────────


def test_random_is_a_sort_key():
    assert "random" in mod.SORT_KEYS


def test_random_is_offered_on_the_cli():
    args = mod._build_arg_parser().parse_args(
        ["--training-wide-csv", "a", "--wide-csv", "b",
         "--dataset", "D", "--out-dir", "o", "--sort-by", "random"]
    )
    assert args.sort_by == "random"


# ── it really is unordered ────────────────────────────────────────────────


def test_random_order_keeps_every_fly_exactly_once(trials):
    ranked = mod.fly_order(trials, by="binary")
    shuffled = mod.fly_order(trials, by="random", seed="X")
    assert sorted(shuffled) == sorted(ranked)


def test_random_order_is_not_the_vigor_ranking(trials):
    """Guards the failure that would look like success: a 'random' order that
    silently fell through to the ranked one."""
    ranked = mod.fly_order(trials, by="binary")
    if len(ranked) < 3:
        pytest.skip("too few flies for the orders to differ meaningfully")
    seeds = [f"seed-{i}" for i in range(12)]
    assert any(mod.fly_order(trials, by="random", seed=s) != ranked for s in seeds)


def test_random_order_is_not_alphabetical(trials):
    ranked = sorted(mod.fly_order(trials, by="binary"))
    if len(ranked) < 3:
        pytest.skip("too few flies")
    seeds = [f"seed-{i}" for i in range(12)]
    assert any(mod.fly_order(trials, by="random", seed=s) != ranked for s in seeds)


# ── but reproducible ──────────────────────────────────────────────────────


def test_same_seed_gives_the_same_order(trials):
    a = mod.fly_order(trials, by="random", seed="Hex-Training-24-0.1")
    b = mod.fly_order(trials, by="random", seed="Hex-Training-24-0.1")
    assert a == b


def test_different_seeds_generally_differ(trials):
    if len(mod.fly_order(trials, by="binary")) < 3:
        pytest.skip("too few flies")
    orders = {tuple(mod.fly_order(trials, by="random", seed=f"s{i}")) for i in range(12)}
    assert len(orders) > 1


def test_seed_defaults_to_something_stable(trials):
    """No seed must not mean 'new order every run' -- that would silently
    redraw a different figure on every pipeline run."""
    assert mod.fly_order(trials, by="random") == mod.fly_order(trials, by="random")


# ── no score is invented ──────────────────────────────────────────────────


def test_random_has_no_per_fly_score(trials):
    """A random order has no magnitude behind it; showing one would imply the
    rows were ranked after all."""
    assert mod.fly_scores(trials, by="random") == {}


def test_unknown_sort_key_still_raises(trials):
    with pytest.raises(ValueError, match="unknown sort key"):
        mod.fly_scores(trials, by="nonsense")


# ── the figure says so ────────────────────────────────────────────────────


def test_caption_states_the_order_is_random():
    blurb = mod._SORT_BLURB.get("random", "")
    assert "random" in blurb.lower()
    assert "auc" not in blurb.lower()


def test_training_figure_builds_under_random_order(trials):
    order = mod.fly_order(trials, by="random", seed="D")
    fig, meta = mod.figure_training(
        trials, order, dataset=DATASET, mode=mod.BINARY_MODE, sort_by="random",
        pre_s=10.0, post_s=40.0, bin_s=0.1,
    )
    assert fig is not None
    assert meta.get("fly_scores") in (None, {})
    matplotlib.pyplot.close(fig)


def test_testing_figure_builds_under_random_order():
    from test_binarized_per_rasters import _testing_frame

    test = mod.build_trials(_testing_frame(), dataset=DATASET, trial_type="testing")
    order = mod.fly_order(test, by="random", seed="D")
    fig, meta = mod.figure_testing(
        test, order, dataset=DATASET, mode=mod.BINARY_MODE, sort_by="random",
        scores=None, pre_s=10.0, post_s=40.0, bin_s=0.1,
    )
    assert fig is not None
    matplotlib.pyplot.close(fig)


# ── the order CSV must survive a scoreless order ──────────────────────────


def test_order_rows_build_without_scores(trials):
    """Regression: the fly_order CSV indexed ``scores[fly]`` unconditionally and
    raised KeyError on the first randomly-ordered cohort, AFTER both figures had
    already been written -- so the run looked half-successful."""
    order = mod.fly_order(trials, by="random", seed="D")
    rows = mod.fly_order_rows(order, {})
    assert list(rows["fly_id"]) == list(order)
    assert list(rows["rank"]) == list(range(1, len(order) + 1))
    assert rows["mean_training_odor_fraction"].isna().all()


def test_order_rows_keep_scores_when_ranked(trials):
    order = mod.fly_order(trials, by="binary")
    scores = mod.fly_scores(trials, by="binary")
    rows = mod.fly_order_rows(order, scores)
    assert list(rows["mean_training_odor_fraction"]) == [scores[f] for f in order]


def test_order_rows_tolerate_a_partial_score_map(trials):
    order = mod.fly_order(trials, by="binary")
    scores = mod.fly_scores(trials, by="binary")
    scores.pop(order[0])
    rows = mod.fly_order_rows(order, scores)
    assert len(rows) == len(order)
