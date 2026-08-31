"""The pretest arm must pair distance files with their angle files.

This is the concrete failure the third arm fixes. On the real
``Hex-Sensitivity-24-0.1/august_28_batch_1_rig_3`` batch, 91 pretest distance
CSVs sit on disk but ``angle_distance_rms_envelope/`` holds only ``testing``
(32) and ``training`` (24) envelopes — not one pretest envelope. Cause: pretest
was categorised as ``testing``, so its trials were collected under
``TESTING_REGEX``, which does not match a ``pretest_N_<odor>`` label. The
distance file fell back to a whole-label key while the angle file fell into the
fallback map under a different key, and the pair was dropped with
"no matching angle file — skipped".

These tests use the real filename shapes from that batch.
"""

from pathlib import Path

from scripts.analysis.envelope_combined import (
    PRETEST_REGEX,
    TESTING_REGEX,
    TRIAL_TYPE_REGEXES,
    _build_trial_configs,
    _collect_distance_entries,
    _index_trials,
)

ODORS = [
    "ethylbutyrate", "isoamylacetate", "linalool", "citral",
    "hexanol", "3-octonol", "benzaldehyde",
]
FLIES = ["fly1", "fly2", "fly3", "fly4"]


def _entries(phase, kind):
    """Angle/distance entries in the (label, path, category) shape _locate_trials returns."""
    out = []
    for idx, odor in enumerate(ODORS, start=1):
        for fly in FLIES:
            label = f"{phase}_{idx}_{odor}"
            stem = f"{label}_{fly}_distances_{fly}_angle_distance_rms_envelope"
            out.append((label, Path(f"/d/{stem}_{kind}.csv"), phase))
    return out


def _pair_all(distance_entries, angle_entries, trial_type, regex):
    """Mirror the pass-1 pairing in combine_fly_dir; return (paired, skipped)."""
    angle_idx, angle_fallback = _index_trials(angle_entries, regex, trial_type)
    dist_idx = _collect_distance_entries(distance_entries, regex, trial_type)
    paired, skipped = 0, 0
    for key, (dist_path, base_key, _slot) in sorted(dist_idx.items()):
        angle_path = None
        for candidate in (key, base_key):
            angle_path = angle_idx.get(candidate)
            if angle_path:
                break
        if angle_path is None:
            angle_path = angle_fallback.get(dist_path.stem.lower())
        if angle_path is None:
            skipped += 1
        else:
            paired += 1
    return paired, skipped


def test_every_pretest_trial_pairs_under_the_pretest_arm():
    dist = _entries("pretest", "distances")
    angle = _entries("pretest", "angle")
    paired, skipped = _pair_all(dist, angle, "pretest", PRETEST_REGEX)
    assert skipped == 0
    assert paired == len(ODORS) * len(FLIES)  # 7 odors x 4 flies = 28


def test_the_testing_arm_alone_pairs_no_pretest_trials():
    """The pre-fix path: pretest labels reach the testing arm and cannot key."""
    dist = _entries("pretest", "distances")
    angle = _entries("pretest", "angle")
    # Category is "pretest" now, so the testing arm sees nothing at all.
    paired, _ = _pair_all(dist, angle, "testing", TESTING_REGEX)
    assert paired == 0


def test_a_sensitivity_fly_processes_all_three_arms_without_loss():
    """pretest_1..7, training_1..6 and testing_1..7 must all survive pairing."""
    dist, angle = [], []
    for phase in ("pretest", "training", "testing"):
        dist += _entries(phase, "distances")
        angle += _entries(phase, "angle")

    configs = _build_trial_configs(angle, dist)
    assert sorted(name for name, _ in configs) == ["pretest", "testing", "training"]

    total_paired, total_skipped = 0, 0
    for trial_type, regex in configs:
        paired, skipped = _pair_all(dist, angle, trial_type, regex)
        assert paired > 0, f"{trial_type} arm paired nothing"
        total_paired += paired
        total_skipped += skipped

    assert total_skipped == 0
    assert total_paired == len(dist)


def test_the_arms_partition_the_trials_with_no_double_counting():
    """A trial must be claimed by exactly one arm, or it is counted twice."""
    dist = []
    for phase in ("pretest", "training", "testing"):
        dist += _entries(phase, "distances")

    claimed = []
    for trial_type, regex in TRIAL_TYPE_REGEXES:
        claimed += [
            path for path, _, _ in _collect_distance_entries(dist, regex, trial_type).values()
        ]
    assert len(claimed) == len(set(claimed)) == len(dist)


# ── discovery: the second hardcoded phase allowlist ───────────────────────


def test_angle_candidate_discovery_includes_pretest_files(tmp_path):
    """_trial_csv_candidates had its OWN training|testing allowlist.

    This is the discovery path _ensure_angle_percentages uses to write
    ``angle_centered_pct``. "pretest" contains neither "training" nor
    "testing", so every pretest file was dropped here — before any angle was
    computed. _locate_trials still found the pretest DISTANCE files, so the
    pretest arm existed but had no angle partner and every pair was dropped
    with "no matching angle file — skipped". Net effect on the real
    Hex-Sensitivity batch: 28 pretest trials in, 0 envelopes out.
    """
    from scripts.analysis.envelope_combined import _trial_csv_candidates

    fly = tmp_path / "august_28_batch_1_rig_3"
    for phase in ("pretest", "testing", "training"):
        d = fly / f"august_28_batch_1_{phase}_1"
        d.mkdir(parents=True)
        (d / f"august_28_batch_1_{phase}_1_fly1_distances.parquet").write_bytes(b"x")

    names = [p.name for p in _trial_csv_candidates(fly, ("*fly*_distances.parquet",))]
    assert any("pretest" in n for n in names), names
    assert any("testing" in n for n in names)
    assert any("training" in n for n in names)
    assert len(names) == 3


def test_angle_candidate_discovery_still_rejects_non_trial_files(tmp_path):
    """The allowlist exists to keep stray tables out; it must still do that."""
    from scripts.analysis.envelope_combined import _trial_csv_candidates

    fly = tmp_path / "batch"
    fly.mkdir()
    (fly / "calibration_fly1_distances.parquet").write_bytes(b"x")
    (fly / "batch_pretest_1_fly1_distances.parquet").write_bytes(b"x")

    names = [p.name for p in _trial_csv_candidates(fly, ("*fly*_distances.parquet",))]
    assert names == ["batch_pretest_1_fly1_distances.parquet"]


def test_every_phase_token_is_recognised_by_both_discovery_paths(tmp_path):
    """The two allowlists must agree, or one silently starves the other."""
    from scripts.analysis.envelope_combined import TRIAL_TYPE_REGEXES, _trial_csv_candidates

    fly = tmp_path / "batch"
    fly.mkdir()
    for phase, _ in TRIAL_TYPE_REGEXES:
        (fly / f"batch_{phase}_1_fly1_distances.parquet").write_bytes(b"x")

    found = " ".join(p.name for p in _trial_csv_candidates(fly, ("*fly*_distances.parquet",)))
    for phase, _ in TRIAL_TYPE_REGEXES:
        assert f"_{phase}_" in found, f"{phase} dropped at angle-candidate discovery"


# ── the THIRD hardcoded phase allowlist: the wide-build glob ──────────────


def test_wide_build_glob_finds_pretest_envelopes(tmp_path):
    """_find_trial_csvs globbed only ``*testing*`` / ``*training*``.

    Combine now writes pretest envelopes correctly, but this glob never matched
    ``pretest_1_ethylbutyrate_..._angle_distance_rms_envelope.csv``, so the wide
    table came out with 0 pretest rows, the pretest subset CSV was header-only,
    the scorer logged "Extra data table is empty", and no pre-vs-post figure
    was ever drawn. Third phase allowlist of its own kind in this pipeline.
    """
    from scripts.analysis.envelope_exports import _find_trial_csvs

    fly = tmp_path / "august_28_batch_1_rig_3"
    rms = fly / "RMS_calculations"
    rms.mkdir(parents=True)
    for phase, odor in (("pretest", "ethylbutyrate"), ("testing", "hexanol"),
                        ("training", "hexanol")):
        name = f"{phase}_1_{odor}_fly1_distances_fly1_angle_distance_rms_envelope.csv"
        (rms / name).write_text("frame,dir_val\n0,1\n")

    found = sorted(p.name for p in _find_trial_csvs(fly))
    assert len(found) == 3, found
    assert any(n.startswith("pretest_") for n in found), found


def test_wide_build_glob_covers_every_known_phase(tmp_path):
    """Derived from the same phase list combine uses, so they cannot drift."""
    from scripts.analysis.envelope_combined import TRIAL_TYPE_REGEXES
    from scripts.analysis.envelope_exports import TRIAL_PHASES, _find_trial_csvs

    # The two modules keep their own phase lists; they must agree or one
    # silently starves the other.
    assert set(TRIAL_PHASES) == {name for name, _ in TRIAL_TYPE_REGEXES}

    fly = tmp_path / "batch"
    rms = fly / "RMS_calculations"
    rms.mkdir(parents=True)
    for phase, _ in TRIAL_TYPE_REGEXES:
        (rms / f"{phase}_1_odor_fly1_distances_fly1_angle_distance_rms_envelope.csv"
         ).write_text("frame,dir_val\n0,1\n")

    found = " ".join(p.name for p in _find_trial_csvs(fly))
    for phase, _ in TRIAL_TYPE_REGEXES:
        assert f"{phase}_1_" in found, f"{phase} missed by the wide-build glob"


def test_wide_build_glob_still_ignores_unrelated_csvs(tmp_path):
    from scripts.analysis.envelope_exports import _find_trial_csvs

    fly = tmp_path / "batch"
    rms = fly / "RMS_calculations"
    rms.mkdir(parents=True)
    (rms / "fly_norm_metadata.csv").write_text("a\n1\n")
    (rms / "pretest_1_odor_fly1_distances_fly1_angle_distance_rms_envelope.csv"
     ).write_text("frame,dir_val\n0,1\n")

    found = [p.name for p in _find_trial_csvs(fly)]
    assert found == ["pretest_1_odor_fly1_distances_fly1_angle_distance_rms_envelope.csv"]


# ── the FOURTH allowlist: build_wide_csv's own discovery ──────────────────


def test_build_wide_csv_discovery_finds_pretest_envelopes(tmp_path):
    """envelope_combined has its OWN _find_trial_csvs, and build_wide_csv calls
    THAT one — not the identically-named function in envelope_exports.

    Fixing only the envelope_exports copy left the live path untouched: the
    wide table still got 0 pretest rows. Two same-named functions in two
    modules is exactly how the first fix missed.
    """
    from scripts.analysis.envelope_combined import _find_trial_csvs

    fly = tmp_path / "august_28_batch_1_rig_3"
    out = fly / "angle_distance_rms_envelope"
    out.mkdir(parents=True)
    for phase, odor in (("pretest", "linalool"), ("testing", "hexanol"),
                        ("training", "hexanol")):
        (out / f"{phase}_1_{odor}_fly1_distances_fly1_angle_distance_rms_envelope.csv"
         ).write_text("frame,dir_val\n0,1\n")

    found = sorted(p.name for p in _find_trial_csvs(fly))
    assert len(found) == 3, found
    assert any(n.startswith("pretest_") for n in found), found


def test_build_wide_csv_discovery_still_prefers_parquet_over_csv(tmp_path):
    """The dedupe relies on every parquet pattern running before every csv
    pattern; deriving the patterns must not interleave them."""
    from scripts.analysis.envelope_combined import _find_trial_csvs

    fly = tmp_path / "batch"
    out = fly / "angle_distance_rms_envelope"
    out.mkdir(parents=True)
    stem = "pretest_1_linalool_fly1_distances_fly1_angle_distance_rms_envelope"
    (out / f"{stem}.parquet").write_bytes(b"x")
    (out / f"{stem}.csv").write_text("frame,dir_val\n0,1\n")

    found = [p for p in _find_trial_csvs(fly)]
    assert len(found) == 1, [p.name for p in found]
    assert found[0].suffix == ".parquet"


def test_both_modules_discover_the_same_phases(tmp_path):
    """The two same-named functions must agree, or one starves the other."""
    from scripts.analysis.envelope_combined import _find_trial_csvs as combined_find
    from scripts.analysis.envelope_exports import _find_trial_csvs as exports_find
    from scripts.analysis.envelope_combined import TRIAL_TYPE_REGEXES

    def _make(root, sub):
        d = root / sub
        d.mkdir(parents=True)
        for phase, _ in TRIAL_TYPE_REGEXES:
            (d / f"{phase}_1_odor_fly1_distances_fly1_angle_distance_rms_envelope.csv"
             ).write_text("frame,dir_val\n0,1\n")
        return root

    a = _make(tmp_path / "a", "angle_distance_rms_envelope")
    b = _make(tmp_path / "b", "RMS_calculations")

    phases_a = {n.split("_")[0] for n in (p.name for p in combined_find(a))}
    phases_b = {n.split("_")[0] for n in (p.name for p in exports_find(b))}
    assert phases_a == phases_b == {name for name, _ in TRIAL_TYPE_REGEXES}
