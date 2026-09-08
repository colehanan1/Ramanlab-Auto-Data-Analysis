"""The demo copy: raw inputs only, additive, never into the source trees."""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "pipeline"))
import make_geometry_demo_copy as MC  # noqa: E402


def _secured_batch(tmp_path):
    src = tmp_path / "secured" / "Hex-Sensitivity-24-0.1" / "september_02_batch_1_rig_3"
    src.mkdir(parents=True)
    stem = "output_september_02_batch_1_testing_9_LightOnly_20260902_132557"
    for ext in (".mp4", ".csv", ".txt"):
        (src / f"{stem}{ext}").write_text("x")
    (src / f"sensors_{stem}.csv").write_text("x")
    (src / "session_metadata.txt").write_text("Host: Flybehavior3\n")
    (src / "odor_vials.csv").write_text("a\n")
    (src / "fly1_global_distance_stats_class_0.json").write_text("{}")
    (src / stem).mkdir()                                   # the stray directory named like an output file
    (src / "september_02_batch_1_testing_9").mkdir()
    (src / "september_02_batch_1_testing_9" / "x_fly1_distances.parquet").write_text("x")
    (src / "RMS_calculations").mkdir()
    return src


def test_plan_files_lists_only_raw_input_files(tmp_path):
    src = _secured_batch(tmp_path)
    names = sorted(p.name for p in MC.plan_files(src))
    assert names == sorted([
        "output_september_02_batch_1_testing_9_LightOnly_20260902_132557.mp4",
        "output_september_02_batch_1_testing_9_LightOnly_20260902_132557.csv",
        "output_september_02_batch_1_testing_9_LightOnly_20260902_132557.txt",
        "sensors_output_september_02_batch_1_testing_9_LightOnly_20260902_132557.csv",
        "session_metadata.txt", "odor_vials.csv"])


def test_rsync_command_is_additive_and_file_only(tmp_path):
    cmd = MC.build_rsync_command(tmp_path / "s", tmp_path / "d", dry_run=True, overwrite=False)
    assert cmd[0] == "rsync" and cmd[-2].endswith("/s/") and cmd[-1].endswith("/d/")
    assert not any(a.startswith("--delete") for a in cmd)
    assert "--dry-run" in cmd and "--ignore-existing" in cmd
    assert cmd.index("--exclude=*/") < cmd.index("--include=output_*.mp4") < cmd.index("--exclude=*")
    live = MC.build_rsync_command(tmp_path / "s", tmp_path / "d", dry_run=False, overwrite=True)
    assert "--dry-run" not in live and "--ignore-existing" not in live


def test_validate_paths_refuses_source_trees_as_destinations(tmp_path):
    src = _secured_batch(tmp_path)
    MC.validate_paths(src, tmp_path / "flys_Abdomen" / "Hex" / "b_copy")
    with pytest.raises(ValueError):
        MC.validate_paths(src, src.parent / "b_copy")               # inside the source dataset
    with pytest.raises(ValueError):
        MC.validate_paths(src, Path("/securedstorage/DATAsec/cole/Data-secured-New/x/y_copy"))
    with pytest.raises(ValueError):
        MC.validate_paths(src, Path("/home/ramanlab/Documents/cole/Data/flys_New/x/y_copy"))
    with pytest.raises(FileNotFoundError):
        MC.validate_paths(tmp_path / "missing", tmp_path / "flys_Abdomen" / "x")
    # traversal and symlinks are resolved before the check, even for a not-yet-existing tail
    with pytest.raises(ValueError):
        MC.validate_paths(src, tmp_path / "flys_Abdomen" / ".." / "secured" / "Hex-Sensitivity-24-0.1" / "b_copy")
    link = tmp_path / "link_to_secured_dataset"
    link.symlink_to(src.parent)
    with pytest.raises(ValueError):
        MC.validate_paths(src, link / "new_copy")


@pytest.mark.skipif(subprocess.run(["which", "rsync"], capture_output=True).returncode != 0, reason="rsync missing")
def test_main_dry_run_copies_nothing_and_run_copies_raw_inputs_only(tmp_path, capsys):
    src = _secured_batch(tmp_path)
    dest_base = tmp_path / "flys_Abdomen"
    args = ["--dataset", "Hex-Sensitivity-24-0.1", "--batch", "september_02_batch_1_rig_3",
            "--source-base", str(tmp_path / "secured"), "--dest-base", str(dest_base)]
    assert MC.main(args) == 0
    dst = dest_base / "Hex-Sensitivity-24-0.1" / "september_02_batch_1_rig_3_copy"
    assert not dst.exists() or not any(dst.iterdir())
    assert MC.main(args + ["--run"]) == 0
    got = sorted(p.name for p in dst.iterdir())
    assert got == sorted(p.name for p in MC.plan_files(src))
    assert not (dst / "RMS_calculations").exists() and not (dst / "september_02_batch_1_testing_9").exists()
    (dst / "odor_vials.csv").write_text("edited\n")
    assert MC.main(args + ["--run"]) == 0
    assert (dst / "odor_vials.csv").read_text() == "edited\n", "existing files are never overwritten by default"


def test_all_mode_plans_every_batch_of_every_dataset_with_original_names(tmp_path):
    src_base = tmp_path / "secured"
    for ds, batches in (("Hex-Sensitivity-24-0.1", ("august_28_batch_1_rig_3", "september_02_batch_1_rig_3")),
                        ("3Oct-Control-24-0.1", ("july_01_batch_1",))):
        for b in batches:
            d = src_base / ds / b
            d.mkdir(parents=True)
            (d / f"output_{b}_testing_1_Hexanol_20260901_120000.mp4").write_text("x")
            (d / "session_metadata.txt").write_text("Host: x\n")
    (src_base / "zLight" / "no_videos_here").mkdir(parents=True)          # no output_*.mp4 -> skipped
    (src_base / "stray_file.txt").write_text("x")
    jobs = MC.plan_all(src_base, tmp_path / "flys_Abdomen", suffix="")
    assert [(j.src.parent.name, j.src.name, j.dst.name) for j in jobs] == [
        ("3Oct-Control-24-0.1", "july_01_batch_1", "july_01_batch_1"),
        ("Hex-Sensitivity-24-0.1", "august_28_batch_1_rig_3", "august_28_batch_1_rig_3"),
        ("Hex-Sensitivity-24-0.1", "september_02_batch_1_rig_3", "september_02_batch_1_rig_3")]
    only = MC.plan_all(src_base, tmp_path / "flys_Abdomen", suffix="", datasets=["Hex-Sensitivity-24-0.1"])
    assert len(only) == 2 and all(j.dst.parent.name == "Hex-Sensitivity-24-0.1" for j in only)


@pytest.mark.skipif(subprocess.run(["which", "rsync"], capture_output=True).returncode != 0, reason="rsync missing")
def test_main_all_copies_every_batch(tmp_path):
    src_base = tmp_path / "secured"
    for ds, b in (("A-Set", "b1"), ("B-Set", "b2")):
        d = src_base / ds / b
        d.mkdir(parents=True)
        (d / f"output_{b}_testing_1_X_20260901_120000.mp4").write_text("x")
        (d / f"output_{b}_testing_1_X_20260901_120000.csv").write_text("x")
        (d / "derived_dir").mkdir()
    dest = tmp_path / "flys_Abdomen"
    assert MC.main(["--all", "--source-base", str(src_base), "--dest-base", str(dest)]) == 0
    assert not dest.exists() or not list(dest.rglob("*.mp4")), "dry run copies nothing"
    assert MC.main(["--all", "--source-base", str(src_base), "--dest-base", str(dest), "--run"]) == 0
    assert sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*.mp4")) == [
        "A-Set/b1/output_b1_testing_1_X_20260901_120000.mp4", "B-Set/b2/output_b2_testing_1_X_20260901_120000.mp4"]
    assert not (dest / "A-Set" / "b1" / "derived_dir").exists()
