"""Only redraw what changed: a figure is keyed by the rows it is drawn from plus the code.

A run rewrites the wide tables whenever any dataset gains a fly, so file times
cannot say which figures are stale (user 2026-11-09: "just those that have updates
or new files for new flys").
"""

from __future__ import annotations

import ast
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import scripts.analysis.pretest_vs_test_traces as ptt  # noqa: E402
from fbpipe.utils import figure_cache as fc  # noqa: E402


def _frame():
    return pd.DataFrame({"fly": ["b", "a"], "fly_number": [1, 2], "v": [0.5, np.nan]})


def test_frame_digest_ignores_row_order():
    df = _frame()
    assert fc.frame_digest(df, ["fly"]) == fc.frame_digest(df.iloc[::-1], ["fly"])


def test_frame_digest_moves_with_a_value_a_new_row_or_a_column():
    df = _frame()
    base = fc.frame_digest(df, ["fly"])
    assert fc.frame_digest(df.assign(v=[0.6, np.nan]), ["fly"]) != base
    assert fc.frame_digest(pd.concat([df, df.iloc[:1]]), ["fly"]) != base
    assert fc.frame_digest(df.rename(columns={"v": "w"}), ["fly"]) != base


def test_frame_digest_ignores_the_order_of_rows_that_share_a_key():
    """A stable sort on ``sort_by`` left tied rows in the order they came in."""
    df = pd.DataFrame({"fly": ["a", "a", "b"], "v": [1.0, 2.0, 3.0]})
    assert fc.frame_digest(df, ["fly"]) == fc.frame_digest(df.iloc[[1, 0, 2]], ["fly"])


def _times(unit="ns"):
    t = pd.to_datetime(["2026-11-09 10:00:00", "2026-11-09 10:00:00.5", None], format="ISO8601")
    return pd.Series(t).astype(f"datetime64[{unit}]")


def _typed():
    """A table as pandas 2.3.3 types it: object text with None, int64, datetime64[ns]."""
    return pd.DataFrame({"fly": pd.Series(["b", "a", None], dtype=object), "n": [1, 2, 3],
                         "v": [0.5, np.nan, -0.0], "ok": [True, False, True],
                         "t": _times("ns")})


def test_frame_digest_ignores_the_dtypes_pandas_picked():
    """pandas 3.0.2 types text ``str`` with NaN for missing and builds datetimes in [us]
    where 2.3.3 gives object with None and [ns]: the same values, the same table. So is
    an int column widened to float, or a nullable one."""
    df = _typed()
    want = fc.frame_digest(df, ["fly"])
    same = [
        df.assign(fly=pd.Series(["b", "a", None],
                                dtype=pd.StringDtype("python", na_value=np.nan))),
        df.assign(fly=pd.Series(["b", "a", None], dtype="string")),
        df.assign(n=df["n"].astype(float)),
        df.assign(n=df["n"].astype("Int64")),
        df.assign(v=pd.array([0.5, None, -0.0], dtype="Float64")),
        df.assign(v=[0.5, np.nan, 0.0]),  # -0.0 == 0.0
        df.assign(ok=df["ok"].astype("boolean")),
        df.assign(ok=df["ok"].astype(float)),
        df.assign(t=_times("us")),
        df.assign(t=_times("ms")),
    ]
    for other in same:
        assert fc.frame_digest(other, ["fly"]) == want, dict(other.dtypes)
    utc = df.assign(t=_times("ns").dt.tz_localize("UTC"))
    assert (fc.frame_digest(utc.assign(t=_times("us").dt.tz_localize("UTC")), ["fly"])
            == fc.frame_digest(utc, ["fly"]))
    assert np.signbit(df["v"].to_numpy()[2])  # -0.0 is evened out in a copy, not here


def test_frame_digest_still_moves_with_every_kind_of_value():
    df = _typed()
    want = fc.frame_digest(df, ["fly"])
    changed = [
        df.assign(t=df["t"] + pd.Timedelta(microseconds=1)),
        df.assign(t=df["t"].dt.tz_localize("UTC")),
        df.assign(fly=["b", "a", "None"]),  # a missing value is not the text "None",
        df.assign(fly=["b", "a", "nan"]),  # nor "nan",
        df.assign(v=[0.5, 0.0, -0.0]),  # nor 0
        df.assign(n=[1, 2, 4]),
        df.assign(ok=[True, True, True]),
    ]
    for other in changed:
        assert fc.frame_digest(other, ["fly"]) != want, other
    # Values that trade places with a column of the other kind make another table.
    assert (fc.frame_digest(pd.DataFrame({"x": [1.0], "y": ["s"]}))
            != fc.frame_digest(pd.DataFrame({"x": ["s"], "y": [1.0]})))


def test_frame_digest_is_one_value_under_pandas_2_and_3():
    """The suite runs under yolo-env (pandas 2.3.3) and ~/.local (3.0.2), which type this
    table differently; pinning its digest makes both agree, or every figure would redraw
    whenever a run used the other one. Changing the digest on purpose redraws
    everything once: update the value here."""
    df = pd.DataFrame({
        "fly": ["b", "a", None], "n": [1, 2, 3], "v": [0.5, np.nan, -0.0],
        "ok": [True, False, True],
        "t": pd.to_datetime(["2026-11-09 10:00:00", "2026-11-09 10:00:00.5", None],
                            format="ISO8601"),
    })
    assert (fc.frame_digest(df, ["fly"])
            == "148bef555b7e015c9094f64983688064a60a45dad4910533d45f1dc3f564d674")


def test_frame_digest_takes_seconds_on_a_full_wide_table():
    """It is called on wide tables: about 12,000 float columns and up to 2,000 rows."""
    rng = np.random.default_rng(0)
    wide = pd.DataFrame(rng.random((2000, 12000)),
                        columns=[f"dir_val_{i}" for i in range(12000)])
    wide.insert(0, "fly", [f"fly{i % 40}" for i in range(2000)])
    start = time.perf_counter()
    fc.frame_digest(wide, ["fly"])
    assert time.perf_counter() - start < 5


def _code_tree(root):
    analysis = root / "scripts" / "analysis"
    utils = root / "src" / "fbpipe" / "utils"
    analysis.mkdir(parents=True)
    utils.mkdir(parents=True)
    for init in (root / "src" / "fbpipe" / "__init__.py", utils / "__init__.py"):
        init.write_text("")
    files = {
        "entry": analysis / "entry.py",
        "lib": analysis / "lib.py",
        "child": analysis / "child.py",
        "grandchild": analysis / "grandchild.py",
        "helper": utils / "helper.py",
        "relative": utils / "sibling.py",
        "unrelated": analysis / "unrelated.py",
    }
    files["entry"].write_text(
        "from fbpipe.utils import helper\n"
        "import scripts.analysis.lib as lib\n"
        "CMD = ['python', 'scripts/analysis/child.py']\n")
    files["lib"].write_text("Y = 2\n")
    files["child"].write_text("from scripts.analysis import grandchild\n")
    files["grandchild"].write_text("Z = 3\n")
    files["helper"].write_text("from .sibling import W\n")
    files["relative"].write_text("W = 1\n")
    files["unrelated"].write_text("U = 4\n")
    return files


def test_code_digest_ignores_scripts_the_step_never_runs(tmp_path):
    """2026-11-09: three new live_eye_* scripts by another session made every
    cohort redraw. Only code a step imports or launches may count."""
    files = _code_tree(tmp_path)
    base = fc.code_digest([files["entry"]], root=tmp_path)
    files["unrelated"].write_text("U = 5\n")
    (files["unrelated"].parent / "brand_new.py").write_text("V = 1\n")
    assert fc.code_digest([files["entry"]], root=tmp_path) == base


def test_code_digest_follows_imports_and_launched_scripts(tmp_path):
    files = _code_tree(tmp_path)
    for name in ("entry", "lib", "child", "grandchild", "helper", "relative"):
        before = fc.code_digest([files["entry"]], root=tmp_path)
        original = files[name].read_text()
        files[name].write_text(original + "# changed\n")
        assert fc.code_digest([files["entry"]], root=tmp_path) != before, name
        files[name].write_text(original)


def _closure(entry, root):
    root = root.resolve()
    return {str(p.relative_to(root)) for p in fc.code_files([entry], root=root)}


def test_a_py_string_counts_only_inside_a_command(tmp_path):
    """2026-11-09: autofreeze's tuple of provenance globs names envelope_combined.py, so
    it and its imports were in the closure, and editing them redrew every figure. Only
    a list or tuple led by a Python interpreter launches a script."""
    files = _code_tree(tmp_path)
    files["entry"].write_text(
        "from fbpipe.utils import helper\n"
        "_CODE_GLOBS = ('src/fbpipe/pipeline.py', 'scripts/analysis/child.py')\n"
        "_SKIP = ['__init__.py', 'lib.py']\n"
        "NOTE = 'grandchild.py'\n"
        "SHELL = ['bash', 'scripts/analysis/unrelated.py']\n"
        "LINE = 'python scripts/analysis/unrelated.py'\n")
    assert _closure(files["entry"], tmp_path) == {
        "scripts/analysis/entry.py", "src/fbpipe/__init__.py", "src/fbpipe/utils/__init__.py",
        "src/fbpipe/utils/helper.py", "src/fbpipe/utils/sibling.py"}


def test_each_way_a_command_names_its_script(tmp_path):
    """A bare name is in scripts/analysis, a path with a folder is from the repo root, and
    a launched script's imports count too, even one inside a function."""
    root = tmp_path / "repo"
    analysis, tools = root / "scripts" / "analysis", root / "scripts" / "tools"
    analysis.mkdir(parents=True)
    tools.mkdir(parents=True)
    for name in ("a", "b", "c", "d", "e", "same"):
        (analysis / f"{name}.py").write_text("")
    for name in ("u", "same"):
        (tools / f"{name}.py").write_text("")
    (tools / "t.py").write_text("def main():\n    from scripts.analysis import e\n")
    outside = tmp_path / "outside.py"
    outside.write_text("")
    entry = analysis / "entry.py"
    entry.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "PY = sys.executable\n"
        "def commands(REPO_ROOT, python_exec, PYTHON, a='scripts/analysis/'):\n"
        "    return [\n"
        "        [PY, a + 'a.py', '--out-dir', 'x'],\n"
        "        [PY, REPO_ROOT / 'scripts/analysis/b.py'],\n"
        "        (sys.executable, str(REPO_ROOT / 'scripts' / 'tools' / 't.py')),\n"
        "        [str(Path(python_exec)), 'scripts/tools/u.py'],\n"
        "        [PYTHON, f'{a}c.py'],\n"
        "        ['python3', 'same.py'],\n"
        f"        [PY, '../outside.py', {str(outside)!r}],\n"
        "    ]\n")
    assert _closure(entry, root) == {
        "scripts/analysis/entry.py", "scripts/analysis/a.py", "scripts/analysis/b.py",
        "scripts/analysis/c.py", "scripts/analysis/e.py", "scripts/analysis/same.py",
        "scripts/tools/t.py", "scripts/tools/u.py"}


def test_the_window_build_counts_every_script_it_launches():
    """A figure step missing from the closure could change without a redraw. The steps
    are read from start_window_report.run_figures itself, so a new one is covered."""
    analysis = fc.REPO_ROOT / "scripts" / "analysis"
    tree = ast.parse((analysis / "start_window_report.py").read_text())
    run_figures = next(node for node in tree.body
                       if isinstance(node, ast.FunctionDef) and node.name == "run_figures")
    steps = {node.value for node in ast.walk(run_figures)
             if isinstance(node, ast.Constant) and isinstance(node.value, str)
             and node.value.endswith(".py")}
    assert "pretest_vs_test_traces.py" in steps, steps
    closure = set(fc.code_files([analysis / "sensitivity_window_results.py"]))
    for name in {"start_window_report.py", "time_of_day_trend.py", *steps}:
        script = fc.REPO_ROOT / name if "/" in name else analysis / name
        assert script.resolve() in closure, name


def test_file_digest_is_empty_for_a_missing_file(tmp_path):
    assert fc.file_digest(tmp_path / "nope") == ""
    (tmp_path / "f").write_text("1")
    assert fc.file_digest(tmp_path / "f") != ""


def test_cache_round_trips_and_forgets(tmp_path):
    path = tmp_path / "cache.json"
    c = fc.FigureCache(path)
    assert not c.is_current("k", "f1")
    c.record("k", "f1")
    c.save()
    again = fc.FigureCache(path)
    assert again.is_current("k", "f1") and not again.is_current("k", "f2")
    again.forget("k")
    assert not again.is_current("k", "f1")


def test_a_corrupt_cache_just_means_a_full_rebuild(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text("{not json")
    assert not fc.FigureCache(path).is_current("k", "f1")


# ── per-fly traces: only new or changed flies are drawn ───────────────────


def _pair(fly="b1", number="1", score=2, value=1.0):
    t = ptt.Trace(env=np.full(10, value), fps=40.0, odor_on_s=30.0, odor_off_s=60.0,
                  trial_label="testing_1_hexanol", score=score)
    return ptt.FlyPair(dataset="Hex-Sensitivity-1-0.1", fly=fly, fly_number=number,
                       odors=["Hexanol"], pre={"Hexanol": t}, post={"Hexanol": t},
                       trained_odor="Hexanol")


def _drawn(monkeypatch):
    calls = []

    def fake(pair, style):
        calls.append(pair.fly_number)
        return plt.figure(figsize=(1, 1))

    monkeypatch.setattr(ptt, "build_figure", fake)
    return calls


STYLE = ptt.TraceStyle()


def test_only_a_new_fly_is_drawn(tmp_path, monkeypatch):
    calls = _drawn(monkeypatch)
    flies = [_pair(number="1"), _pair(number="2")]
    assert ptt.render_changed(flies, tmp_path, STYLE, "code") == (2, 0)
    calls.clear()
    assert ptt.render_changed([*flies, _pair(number="3")], tmp_path, STYLE, "code") == (1, 2)
    assert calls == ["3"]


def test_a_changed_score_or_trace_redraws_only_that_fly(tmp_path, monkeypatch):
    calls = _drawn(monkeypatch)
    ptt.render_changed([_pair(number="1"), _pair(number="2")], tmp_path, STYLE, "code")
    calls.clear()
    ptt.render_changed([_pair(number="1", score=3), _pair(number="2")], tmp_path, STYLE, "code")
    assert calls == ["1"]
    calls.clear()
    ptt.render_changed([_pair(number="1", score=3), _pair(number="2", value=2.0)], tmp_path,
                       STYLE, "code")
    assert calls == ["2"]


def test_changed_code_redraws_every_fly(tmp_path, monkeypatch):
    calls = _drawn(monkeypatch)
    flies = [_pair(number="1"), _pair(number="2")]
    ptt.render_changed(flies, tmp_path, STYLE, "code-v1")
    calls.clear()
    ptt.render_changed(flies, tmp_path, STYLE, "code-v2")
    assert sorted(calls) == ["1", "2"]


def test_a_deleted_png_is_redrawn(tmp_path, monkeypatch):
    calls = _drawn(monkeypatch)
    pair = _pair()
    ptt.render_changed([pair], tmp_path, STYLE, "code")
    ptt.output_path(tmp_path, pair).unlink()
    calls.clear()
    assert ptt.render_changed([pair], tmp_path, STYLE, "code") == (1, 0)


def test_a_trace_moving_between_pre_and_post_redraws_the_fly(tmp_path, monkeypatch):
    """Each phase's traces were hashed in turn without the phase's name, so a trace that
    moved from the pre-test to the post-training test left the fingerprint as it was."""
    calls = _drawn(monkeypatch)
    pair = _pair()
    t = pair.pre["Hexanol"]
    ptt.render_changed([replace(pair, post={})], tmp_path, STYLE, "code")
    calls.clear()
    moved = replace(pair, pre={}, post={"Hexanol": t})
    assert ptt.render_changed([moved], tmp_path, STYLE, "code") == (1, 0)
    # Two odors split between the phases are not both in one phase either.
    split = replace(pair, odors=["Hexanol", "Octanol"], pre={"Hexanol": t},
                    post={"Octanol": t})
    one = replace(split, pre={"Hexanol": t, "Octanol": t}, post={})
    assert ptt.pair_fingerprint(split, STYLE, "code") != ptt.pair_fingerprint(one, STYLE, "code")
