"""The response threshold θ — one implementation, shared by the whole pipeline.

A PER is scored by asking whether the proboscis-extension envelope rose above a
per-trial threshold estimated from that trial's pre-odor baseline. Everything
downstream depends on it: the ``AUC-*`` columns of the wide table, the binary
rasters, the red line on the trace figures, the scoring filters.

Historically each of those computed θ itself, and they disagreed — symmetric vs
one-sided MAD, k = 2 vs a hardcoded k = 3, per-trial vs fly-level baseline. The
red line on a figure and the AUC quoted beside it came from different rules.
This module is the single definition; callers pass a :class:`ThresholdRule`.

The rule
--------
::

    θ = location + max(k · scale, min_delta)

``location`` and ``scale`` come from the pre-odor baseline, by one of two
estimators.

**Legacy** (``anchor_s is None``) — the shipped behaviour, kept as the default so
nothing changes unless a config asks for it. ``location`` is the median of the
whole baseline and ``scale`` its one-sided upper spread.

**Anchored** (``anchor_s`` set) — the whole-window median assumes the fly held
still for the entire baseline. It often did not, and the two ways that fails push
θ in *opposite* directions, so no value of k repairs both:

* a spontaneous **burst** mid-baseline inflates the spread and lifts θ above the
  very response it should detect;
* a **step** — the fly extends at t ≈ 17 s and simply *holds* there — does the
  same, and additionally leaves the median describing a position the proboscis
  abandoned 13 s before odor onset.

``noise_pctl`` is what decides how much a restless baseline is allowed to widen
the band. At 25 (the shipped value) a single burst still cannot set the scale —
it lives in a minority of windows — but a fly that is genuinely fidgeting or
extending spontaneously before the odor carries a wider band into the odor window
and must clear a higher bar to be believed. Pushing it to 75 or 90 costs accuracy
(0.791, 0.763): past that it stops measuring idle noise and starts measuring the
response itself.

Neither is an outlier a robust statistic absorbs; both are real, seconds-long
stretches of the window. So the two quantities are estimated over different
spans: ``location`` from the last ``anchor_s`` seconds (where the proboscis
actually *is* when the odor arrives) and ``scale`` from a low percentile of
rolling σ (how quiet this fly's tracking is when idle, which a minority of loud
windows cannot set).

Why MAD and not SD
------------------
Standard deviation squares deviations about the *mean*, so a single spontaneous
extension during the baseline moves the centre and inflates the spread — its
breakdown point is zero. The median absolute deviation tolerates up to half the
window being contaminated. ``1.4826`` rescales MAD so that for Gaussian noise
``1.4826 · MAD ≈ σ``, which is what lets ``k`` read as "number of SDs".

Only *upward* deviations feed the spread: a downward dip is the opposite of a
proboscis extension, and letting it widen the band would raise the bar for a real
response.

Units, and why one ``min_delta`` serves every fly
-------------------------------------------------
``dir_val_*`` is normalised against **each fly's own** ``global_min``/
``global_max`` over its whole session, so the envelope is already a percentage of
that fly's full extension range. ``min_delta = 5`` therefore reads as "5 % of this
fly's own range", not 5 pixels; body size, camera distance and rig divide out.

Calibration
-----------
Against the 629 hand-scored trials in ``blinded_video_scores.csv`` the anchored
rule reaches balanced accuracy 0.806 at the shipped settings (k = 3, floor 5,
anchor 5 s, 5 s noise blocks at the 25th percentile), against 0.726 (sens 0.830 /
spec 0.622) for the legacy whole-window rule at k = 2, and 0.773 for legacy at
its own best k = 3.

``std_mult`` decides how much a restless baseline raises theta, because for a
quiet fly ``k * scale`` falls below ``min_delta`` and the floor decides instead.
Overall accuracy is flat across k = 2..6 (0.802-0.807), so k is chosen on the
*active* quartile, where k = 3 is best (0.764, against 0.757 at k = 2 and 0.742
at k = 6). At k = 3 theta rises on 11 % of trials and falls on none — the effect
is deliberately concentrated on flies that were already moving before odor on. The optimum floor is 5
with a bootstrap 95 % CI of [4, 9], and the curve is flat from 4 to 8, so the
choice is not knife-edge.

Two caveats travel with that number. Every hand score is a **testing** trial, so
transfer to training trials is assumed, not measured. And per-cohort optima span
3–13: the low-amplitude cohorts (3Oct-Control-24-0.1, whose training AUC spans
0–52 against 0–539 for Hex-Control-24-0.1) are over-suppressed by a floor of 5
and want something nearer 2–3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "MAD_TO_SIGMA",
    "ThresholdRule",
    "baseline_theta",
    "compute_theta",
    "symmetric_sigma",
    "rolling_baseline",
    "upper_sigma",
]

#: Scales a MAD to a Gaussian sigma, so ``std_mult`` reads as "number of SDs".
MAD_TO_SIGMA = 1.4826

#: Config keys a step may carry. Kept here so both configs and every step that
#: reads them agree on the spelling.
CONFIG_KEYS = (
    "threshold_std_mult",
    "threshold_min_delta",
    "threshold_anchor_s",
    "threshold_noise_block_s",
    "threshold_noise_pctl",
)


def symmetric_sigma(window: Sequence[float] | np.ndarray) -> float:
    """``1.4826 * median(|x - median|)`` -- the ordinary two-sided MAD.

    Kept only so the v1 codebase remains exactly reproducible: ``protocol:
    legacy`` computed its AUC threshold this way, at a hardcoded k = 3. Do not
    reach for it in new code -- a downward dip is the opposite of a proboscis
    extension and should not widen the band. Use :func:`upper_sigma`.
    """
    w = np.asarray(window, dtype=float)
    if w.size == 0:
        return 0.0
    median = float(np.nanmedian(w))
    if not math.isfinite(median):
        return 0.0
    mad = float(np.nanmedian(np.abs(w - median)))
    if not math.isfinite(mad):
        return 0.0
    return MAD_TO_SIGMA * mad


def upper_sigma(window: Sequence[float] | np.ndarray) -> float:
    """One-sided (upper) robust spread of ``window``, scaled to a sigma.

    The median of the *positive* deviations from the median. For symmetric noise
    that is the 75th percentile of the distribution, which is also the ordinary
    MAD, so :data:`MAD_TO_SIGMA` carries over unchanged. Returns 0.0 when nothing
    sits above the median.
    """
    w = np.asarray(window, dtype=float)
    if w.size == 0:
        return 0.0
    baseline = float(np.nanmedian(w))
    if not math.isfinite(baseline):
        return 0.0
    dev = w - baseline
    up = dev[np.isfinite(dev) & (dev > 0.0)]
    if up.size == 0:
        return 0.0
    return MAD_TO_SIGMA * float(np.nanmedian(up))


def baseline_theta(
    window: Sequence[float] | np.ndarray,
    std_mult: float,
    min_delta: float = 0.0,
) -> float:
    """``median(window) + max(std_mult * upper_sigma(window), min_delta)``.

    ``min_delta`` floors how far above the resting median θ may sit. A flat
    baseline drives the spread toward zero and collapses θ onto the median, at
    which point a couple of units of drift during the odor register as a full
    response. It is a ``max``, never a clamp down: a fly that genuinely moves the
    whole time keeps its wider, spread-driven threshold. 0.0 disables it.
    """
    w = np.asarray(window, dtype=float)
    if w.size == 0:
        return math.nan
    baseline = float(np.nanmedian(w))
    if not math.isfinite(baseline):
        return math.nan
    return float(baseline + max(std_mult * upper_sigma(w), float(min_delta)))


def rolling_baseline(
    before: Sequence[float] | np.ndarray,
    fps: float,
    anchor_s: float,
    noise_block_s: float = 5.0,
    noise_pctl: float = 25.0,
) -> tuple[float, float]:
    """``(location, scale)`` for a baseline that is not stationary.

    ``location`` is the median of the last ``anchor_s`` seconds; ``scale`` is the
    ``noise_pctl`` percentile of :func:`upper_sigma` over rolling
    ``noise_block_s`` windows. See the module docstring for why the two are
    estimated over different spans.
    """
    b = np.asarray(before, dtype=float)
    finite = b[np.isfinite(b)]
    if finite.size == 0 or not math.isfinite(fps) or fps <= 0:
        return math.nan, 0.0

    n_anchor = min(max(int(round(anchor_s * fps)), 1), finite.size)
    location = float(np.median(finite[-n_anchor:]))

    n_block = int(round(noise_block_s * fps))
    if n_block < 2 or finite.size < n_block:
        return location, upper_sigma(finite)

    step = max(n_block // 4, 1)
    sigmas = [
        upper_sigma(finite[i : i + n_block])
        for i in range(0, finite.size - n_block + 1, step)
    ]
    scale = float(np.percentile(sigmas, noise_pctl)) if sigmas else 0.0
    return location, scale


def compute_theta(
    env: Sequence[float] | np.ndarray,
    fps: float,
    baseline_until_s: float,
    std_mult: float,
    min_delta: float = 0.0,
    anchor_s: float | None = None,
    noise_block_s: float = 5.0,
    noise_pctl: float = 25.0,
) -> float:
    """θ for one trial, estimated on the pre-command baseline only.

    ``anchor_s = None`` selects the legacy whole-window estimator, so omitting the
    new arguments reproduces the shipped threshold exactly.
    """
    e = np.asarray(env, dtype=float)
    if e.size == 0 or not math.isfinite(fps) or fps <= 0:
        return math.nan

    before_end = min(int(round(baseline_until_s * fps)), e.size)
    if before_end <= 0:
        return math.nan

    before = e[:before_end]
    if anchor_s is None:
        return baseline_theta(before, std_mult, min_delta)

    location, scale = rolling_baseline(
        before, fps, float(anchor_s), noise_block_s, noise_pctl
    )
    if not math.isfinite(location):
        return math.nan
    return float(location + max(std_mult * scale, float(min_delta)))


@dataclass(frozen=True)
class ThresholdRule:
    """The parameters of θ, as one value a pipeline step can carry around.

    Defaults reproduce the shipped legacy behaviour, so a step that has not been
    given the new config keys keeps scoring exactly as it did.
    """

    std_mult: float = 2.0
    min_delta: float = 0.0
    anchor_s: float | None = None
    noise_block_s: float = 5.0
    noise_pctl: float = 25.0
    #: Use the two-sided MAD instead of the one-sided upper spread. Exists only
    #: to reproduce v1 (``protocol: legacy``, k = 3); see :func:`symmetric_sigma`.
    symmetric: bool = False

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | None) -> "ThresholdRule":
        """Read the ``threshold_*`` keys out of a config block, ignoring the rest.

        A key present but null (``threshold_anchor_s:`` with no value, which YAML
        parses as ``None``) means "not set" and must not become ``0.0`` seconds —
        that would silently anchor on an empty window.
        """
        cfg = cfg or {}
        d = cls()

        def num(key: str, fallback: float) -> float:
            v = cfg.get(key, None)
            if v is None:
                return fallback
            try:
                f = float(v)
            except (TypeError, ValueError):
                return fallback
            return f if math.isfinite(f) else fallback

        anchor_raw = cfg.get("threshold_anchor_s", None)
        anchor: float | None
        if anchor_raw is None:
            anchor = None
        else:
            try:
                anchor = float(anchor_raw)
            except (TypeError, ValueError):
                anchor = None
            else:
                if not math.isfinite(anchor) or anchor <= 0:
                    anchor = None

        return cls(
            std_mult=num("threshold_std_mult", d.std_mult),
            min_delta=num("threshold_min_delta", d.min_delta),
            anchor_s=anchor,
            noise_block_s=num("threshold_noise_block_s", d.noise_block_s),
            noise_pctl=num("threshold_noise_pctl", d.noise_pctl),
            symmetric=bool(cfg.get("threshold_symmetric", d.symmetric)),
        )

    @classmethod
    def v1(cls) -> "ThresholdRule":
        """Exactly what the v1 codebase used for the AUC columns.

        Symmetric MAD at a hardcoded k = 3. This is the fallback when a caller
        supplies no rule at all, so ``protocol: legacy`` on a config that carries
        no ``threshold_*`` keys still reproduces v1 byte for byte.
        """
        return cls(std_mult=3.0, symmetric=True)

    def theta(
        self,
        env: Sequence[float] | np.ndarray,
        *,
        fps: float,
        baseline_until_s: float,
    ) -> float:
        """θ for one trial under this rule."""
        return compute_theta(
            env,
            fps,
            baseline_until_s,
            self.std_mult,
            min_delta=self.min_delta,
            anchor_s=self.anchor_s,
            noise_block_s=self.noise_block_s,
            noise_pctl=self.noise_pctl,
        )

    def theta_from_baseline(self, before: Sequence[float] | np.ndarray, *, fps: float) -> float:
        """θ when the caller already sliced the baseline out of the trace."""
        b = np.asarray(before, dtype=float)
        if b.size == 0:
            return math.nan
        if self.symmetric:
            median = float(np.nanmedian(b))
            if not math.isfinite(median):
                return math.nan
            return float(
                median + max(self.std_mult * symmetric_sigma(b), self.min_delta)
            )
        if self.anchor_s is None:
            return baseline_theta(b, self.std_mult, self.min_delta)
        location, scale = rolling_baseline(
            b, fps, float(self.anchor_s), self.noise_block_s, self.noise_pctl
        )
        if not math.isfinite(location):
            return math.nan
        return float(location + max(self.std_mult * scale, self.min_delta))

    def describe(self) -> str:
        """A one-line human-readable form, for figure captions and JSON sidecars."""
        if self.symmetric:
            return (
                f"v1: median(baseline) + max({self.std_mult:g}"
                f"·σ_symmetric, {self.min_delta:g})"
            )
        if self.anchor_s is None:
            return (
                f"legacy: median(baseline) + max({self.std_mult:g}"
                f"·σ_upper, {self.min_delta:g})"
            )
        return (
            f"anchored: median(last {self.anchor_s:g} s) + max({self.std_mult:g}"
            f"·σ_rolling, {self.min_delta:g})"
        )
