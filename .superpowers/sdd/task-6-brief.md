### Task 6: Figure freeze

**Files:**
- Modify: `scripts/analysis/envelope_visuals.py:303-331` (`resolve_dataset_output_dir`), and the figure emit sites
- Test: `tests/test_freeze_figures.py` (create)

**Interfaces:**
- Consumes: `freeze.freeze_flags` (Task 2)
- Produces: `should_skip_frozen_figure(cfg, datasets, *, thawed=(), thaw_all=False) -> bool`

- [ ] **Step 1: Write the failing test**

Create `tests/test_freeze_figures.py`:

```python
"""A figure is skipped only when EVERY contributing dataset is frozen."""

import scripts.analysis.envelope_visuals as ev
from fbpipe.config import DatasetOverride


class _Cfg:
    def __init__(self, overrides):
        self.dataset_overrides = overrides


F = DatasetOverride(freeze_figures=True)
U = DatasetOverride(freeze_figures=False)


def test_single_frozen_dataset_figure_is_skipped():
    assert ev.should_skip_frozen_figure(_Cfg({"A": F}), ["A"]) is True


def test_single_live_dataset_figure_is_drawn():
    assert ev.should_skip_frozen_figure(_Cfg({"A": U}), ["A"]) is False


def test_all_frozen_aggregate_is_skipped():
    assert ev.should_skip_frozen_figure(_Cfg({"A": F, "B": F}), ["A", "B"]) is True


def test_mixed_frozen_and_live_is_DRAWN():
    """The correctness case: frozen Control beside live Training must redraw, or
    adding flies to Training silently fails to appear."""
    assert ev.should_skip_frozen_figure(_Cfg({"A": F, "B": U}), ["A", "B"]) is False


def test_unknown_dataset_counts_as_live():
    assert ev.should_skip_frozen_figure(_Cfg({"A": F}), ["A", "UNKNOWN"]) is False


def test_empty_dataset_set_is_drawn():
    """Never skip on an empty contributor set -- that is 'unknown', not 'all frozen'.
    all([]) is True, which would silently skip every such figure."""
    assert ev.should_skip_frozen_figure(_Cfg({"A": F}), []) is False


def test_freeze_data_alone_does_not_skip_figures():
    """The flags are independent: data:true + figures:false still draws."""
    cfg = _Cfg({"A": DatasetOverride(freeze_data=True, freeze_figures=False)})
    assert ev.should_skip_frozen_figure(cfg, ["A"]) is False


def test_thaw_all_draws_everything():
    assert ev.should_skip_frozen_figure(_Cfg({"A": F}), ["A"], thaw_all=True) is False


def test_thaw_named_dataset_draws():
    assert ev.should_skip_frozen_figure(_Cfg({"A": F}), ["A"], thawed=["A"]) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_freeze_figures.py -v`
Expected: FAIL with `AttributeError: ... has no attribute 'should_skip_frozen_figure'`

- [ ] **Step 3: Implement the guard**

In `scripts/analysis/envelope_visuals.py`, next to `resolve_dataset_output_dir`
(~`:331`):

```python
def should_skip_frozen_figure(cfg, datasets, *, thawed=(), thaw_all=False) -> bool:
    """True when EVERY dataset contributing to a figure is frozen for figures.

    Deliberately NOT "any contributor is frozen". A figure drawn from a frozen
    Control and a live Training must still redraw, or adding flies to Training
    would silently fail to appear in it.

    An empty or unknown contributor set counts as LIVE. Note all([]) is True, so
    an empty set would otherwise skip every such figure -- an empty set means
    "we do not know", not "all frozen".
    """
    from fbpipe.freeze import freeze_flags

    names = [str(d) for d in (datasets or []) if str(d).strip()]
    if not names:
        return False
    for name in names:
        _, freeze_figures = freeze_flags(cfg, name, thawed=thawed, thaw_all=thaw_all)
        if not freeze_figures:
            return False
    return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_freeze_figures.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Apply the guard at the figure emit sites**

Add an early return at each site, using the dataset set that site already
resolves. Do NOT reuse `should_write` (`:334-343`) — it force-returns `True` for
any `reaction_matrix` / `reaction_prediction` path regardless of `overwrite`, so
folding freeze into it would be ignored for exactly those figures.

- `generate_envelope_plots` (`:2024`, savefig `:2677`) — guard per fly, on that fly's dataset.
- `generate_reaction_matrices` (`:1410`, savefig `:1753`) — guard on the contributing set.
- `plot_reaction_rate_bars` (`:1242`) — guard on the contributing set.
- `scripts/analysis/score_summary.py`: `_plot_bar_charts` (`:565`), `_plot_training_vs_control_bars` (`:728`), `_plot_heatmap` (`:859`), `_plot_score_pair` (`:928`).

Each guard logs one line so a skip is visible:

```python
        if should_skip_frozen_figure(cfg, contributing_datasets,
                                     thawed=thawed, thaw_all=thaw_all):
            print(f"[FROZEN] Skipping figure (all contributors frozen): {out_path}")
            return
```

`score_summary.py` runs as a SUBPROCESS (`run_workflows.py:1807-1824`), so it
does not share `settings`. It must load config itself via its existing
`--config` argument and read `dataset_overrides` from there.

- [ ] **Step 6: Honor freeze under `--figures-only`**

`--figures-only` force-sets every figure step to `True` (`:2024-2046`). Do NOT
let that bypass `freeze.figures` — it is the most common way figures are run, so
bypassing there would make the feature nearly inert. `--thaw` remains the way to
force a redraw.

Add to `tests/test_freeze_figures.py`:

```python
def test_figures_only_does_not_bypass_freeze():
    """--figures-only forces figure steps on; it must not un-freeze them."""
    cfg = _Cfg({"A": F})
    assert ev.should_skip_frozen_figure(cfg, ["A"]) is True
```

- [ ] **Step 7: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: PASS, ≥538 passed

- [ ] **Step 8: Commit**

```bash
git add scripts/analysis/envelope_visuals.py scripts/analysis/score_summary.py tests/test_freeze_figures.py
git commit -m "feat(figures): skip a figure only when every contributing dataset is frozen"
```

---

