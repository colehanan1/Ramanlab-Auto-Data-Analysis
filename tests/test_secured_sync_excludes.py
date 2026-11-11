"""A ``.<name>.partial`` left by a killed secure copy is never pushed on.

The file-server rsync and the Box rclone are built in
fbpipe.utils.data_secured_sync (run_workflows._run_data_secured_sync calls them).
"""

from fbpipe.utils import data_secured_sync as dss

PATTERN = ".*.partial"


def test_the_file_server_rsync_excludes_partials(tmp_path):
    cmd = dss.build_file_server_sync_command(tmp_path / "a", tmp_path / "b")
    assert cmd[cmd.index("--exclude") + 1] == PATTERN
    assert cmd[-2].endswith("a/") and cmd[-1].endswith("b/")  # source/dest stay last


def test_the_box_rclone_excludes_partials(tmp_path):
    cmd = dss.build_box_sync_command(tmp_path, remote="box", folder="f")
    assert cmd[cmd.index("--exclude") + 1] == PATTERN


def test_the_excludes_survive_dry_run(tmp_path):
    for cmd in (
        dss.build_file_server_sync_command(tmp_path, tmp_path, dry_run=True, files_from="x"),
        dss.build_box_sync_command(tmp_path, remote="b", folder="", dry_run=True),
    ):
        assert PATTERN in cmd


def test_a_sample_sync_drops_the_exclude_because_rclone_refuses_it_with_files_from(tmp_path):
    """rclone exits 1 when any filter flag is combined with --files-from."""
    cmd = dss.build_box_sync_command(tmp_path, remote="box", folder="f",
                                     files_from=tmp_path / "list.txt")
    assert "--files-from" in cmd and "--exclude" not in cmd


def test_a_sample_never_lists_a_partial(tmp_path):
    (tmp_path / "a.txt").write_text("1")
    (tmp_path / ".b.mp4.partial").write_text("2")
    assert dss.collect_relative_file_sample(tmp_path, 10) == ["a.txt"]
