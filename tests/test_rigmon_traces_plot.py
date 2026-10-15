import math

from matplotlib.image import imread

from rigmon.traces import plot as plot_mod
from rigmon.traces.plot import FLY_COLORS, render_trace
from rigmon.traces.series import TraceSeries


def _series(n=4, k=50):
    t = [i * 0.05 for i in range(k)]
    return TraceSeries(t, {f: [80.0 + (40 if 20 <= i < 30 and f == 2 else 0) for i in range(k)]
                           for f in range(1, n + 1)}, n)


def test_plot_writes_png_of_expected_size(tmp_path):
    out = render_trace(_series(), [(1.0, 1.5, "OFM_I")], tmp_path / "t.png", title="Pi 2 · b · Testing 3")
    img = imread(out)
    assert out.is_file() and img.shape[0] == 500 and img.shape[1] == 1400


def test_plot_handles_three_flies_and_no_odor(tmp_path):
    s = _series(n=3)
    s.flies[3][5] = math.nan
    out = render_trace(s, [], tmp_path / "t.png", title="Pi 1 · b · Training 1")
    assert out.is_file()


def test_plot_handles_empty_series(tmp_path):
    out = render_trace(TraceSeries([], {}, 0), [], tmp_path / "t.png", title="x")
    assert out.is_file()


def test_colors_are_fixed():
    assert FLY_COLORS == ("#0072B2", "#E69F00", "#009E73", "#CC79A7")


def test_plot_module_does_not_use_pyplot():
    assert "plt" not in plot_mod.__dict__ and "matplotlib.pyplot" not in vars(plot_mod).values()
    assert not hasattr(plot_mod, "plt")


def test_plot_renders_from_three_threads_at_once(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(3) as ex:
        outs = list(ex.map(lambda k: render_trace(_series(), [(1.0, 1.5, "OFM_I")],
                                                  tmp_path / f"t{k}.png", title=f"t{k}"), range(3)))
    for o in outs:
        img = imread(o)
        assert img.shape[:2] == (500, 1400)
