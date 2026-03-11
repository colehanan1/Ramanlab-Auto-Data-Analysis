from __future__ import annotations

from pathlib import Path

from fbpipe.utils import data_secured_sync as dss


def test_load_data_secured_sync_config_reads_custom_values(tmp_path: Path) -> None:
    source = tmp_path / "source"
    file_server = tmp_path / "file_server"

    raw_cfg = {
        "backups": {
            "data_secured_sync": {
                "enabled": True,
                "source": str(source),
                "file_server": {
                    "enabled": True,
                    "path": str(file_server),
                },
                "box": {
                    "enabled": True,
                    "remote": "Box-Folder",
                    "folder": "Data-Secured",
                },
            }
        }
    }

    cfg = dss.load_data_secured_sync_config(raw_cfg)

    assert cfg.enabled is True
    assert cfg.source == source.resolve()
    assert cfg.file_server.path == file_server.resolve()
    assert cfg.box.remote == "Box-Folder"
    assert cfg.box.folder == "Data-Secured"


def test_collect_relative_file_sample_prefers_visible_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()

    visible_one = source / "dataset_a" / "trial_1" / "coords.csv"
    visible_two = source / "dataset_b" / "trial_4" / "output.mp4"
    hidden = source / ".cache" / "skip.txt"

    visible_one.parent.mkdir(parents=True)
    visible_two.parent.mkdir(parents=True)
    hidden.parent.mkdir(parents=True)

    visible_one.write_text("a", encoding="utf-8")
    visible_two.write_text("b", encoding="utf-8")
    hidden.write_text("c", encoding="utf-8")

    sample = dss.collect_relative_file_sample(source, 2)

    assert set(sample) == {
        "dataset_a/trial_1/coords.csv",
        "dataset_b/trial_4/output.mp4",
    }


def test_build_file_server_sync_command_is_incremental_and_non_destructive(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "dest"
    files_from = tmp_path / "files_from.txt"

    cmd = dss.build_file_server_sync_command(
        source,
        destination,
        dry_run=True,
        files_from=files_from,
    )

    assert cmd[0] == "rsync"
    assert "--update" in cmd
    assert "--dry-run" in cmd
    assert "--files-from" in cmd
    assert not any(part.startswith("--delete") for part in cmd)
    assert f"{source.resolve()}/" in cmd
    assert f"{destination.resolve()}/" in cmd


def test_build_box_sync_command_uses_copy_not_sync(tmp_path: Path) -> None:
    source = tmp_path / "source"
    files_from = tmp_path / "files_from.txt"

    cmd = dss.build_box_sync_command(
        source,
        remote="Box-Folder",
        folder="Data-Secured",
        dry_run=True,
        files_from=files_from,
    )

    assert cmd[:2] == ["rclone", "copy"]
    assert "--update" in cmd
    assert "--dry-run" in cmd
    assert "--files-from" in cmd
    assert "Box-Folder:Data-Secured" in cmd
    assert not any(part.startswith("--delete") for part in cmd)


def test_sync_data_secured_builds_both_destination_commands(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    file_server = tmp_path / "file_server"
    sample_file = source / "dataset_a" / "trial_1" / "coords.csv"
    sample_file.parent.mkdir(parents=True)
    sample_file.write_text("coords", encoding="utf-8")

    raw_cfg = {
        "backups": {
            "data_secured_sync": {
                "enabled": True,
                "source": str(source),
                "file_server": {
                    "enabled": True,
                    "path": str(file_server),
                },
                "box": {
                    "enabled": True,
                    "remote": "Box-Folder",
                    "folder": "Data-Secured",
                },
            }
        }
    }

    commands: list[list[str]] = []
    files_from_payloads: list[str] = []

    def fake_run_command(label: str, cmd: list[str]) -> bool:
        commands.append(cmd)
        idx = cmd.index("--files-from") + 1
        files_from_payloads.append(Path(cmd[idx]).read_text(encoding="utf-8"))
        return True

    monkeypatch.setattr(dss, "_run_command", fake_run_command)

    results = dss.sync_data_secured(
        raw_cfg,
        sample_files=["dataset_a/trial_1/coords.csv"],
    )

    assert results == {"file_server": True, "box": True}
    assert len(commands) == 2
    assert commands[0][0] == "rsync"
    assert commands[1][:2] == ["rclone", "copy"]
    assert files_from_payloads == [
        "dataset_a/trial_1/coords.csv\n",
        "dataset_a/trial_1/coords.csv\n",
    ]
