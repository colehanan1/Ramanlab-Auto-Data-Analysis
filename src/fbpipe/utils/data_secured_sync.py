from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

DEFAULT_SECURED_SOURCE = Path("/securedstorage/DATAsec/cole/Data-secured")
DEFAULT_FILE_SERVER_PATH = Path("/home/ramanlab/Fileserver/cole/Data-secured")
DEFAULT_BOX_REMOTE = "Box-Folder"
DEFAULT_BOX_FOLDER = "Data-Secured"


@dataclass(frozen=True)
class FileServerSyncConfig:
    enabled: bool = True
    path: Path = DEFAULT_FILE_SERVER_PATH


@dataclass(frozen=True)
class BoxSyncConfig:
    enabled: bool = True
    remote: str = DEFAULT_BOX_REMOTE
    folder: str = DEFAULT_BOX_FOLDER


@dataclass(frozen=True)
class DataSecuredSyncConfig:
    enabled: bool = False
    source: Path = DEFAULT_SECURED_SOURCE
    file_server: FileServerSyncConfig = field(default_factory=FileServerSyncConfig)
    box: BoxSyncConfig = field(default_factory=BoxSyncConfig)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def load_data_secured_sync_config(raw_cfg: Mapping[str, Any]) -> DataSecuredSyncConfig:
    backups_cfg = _as_mapping(raw_cfg.get("backups"))
    sync_cfg = _as_mapping(backups_cfg.get("data_secured_sync"))
    file_server_cfg = _as_mapping(sync_cfg.get("file_server"))
    box_cfg = _as_mapping(sync_cfg.get("box"))

    folder = str(box_cfg.get("folder", DEFAULT_BOX_FOLDER)).strip("/")
    return DataSecuredSyncConfig(
        enabled=bool(sync_cfg.get("enabled", False)),
        source=Path(sync_cfg.get("source", DEFAULT_SECURED_SOURCE)).expanduser().resolve(),
        file_server=FileServerSyncConfig(
            enabled=bool(file_server_cfg.get("enabled", True)),
            path=Path(file_server_cfg.get("path", DEFAULT_FILE_SERVER_PATH)).expanduser().resolve(),
        ),
        box=BoxSyncConfig(
            enabled=bool(box_cfg.get("enabled", True)),
            remote=str(box_cfg.get("remote", DEFAULT_BOX_REMOTE)),
            folder=folder,
        ),
    )


def collect_relative_file_sample(source: Path | str, limit: int) -> list[str]:
    """Return up to *limit* relative file paths for a small verification sync."""

    if limit <= 0:
        return []

    source_path = Path(source).expanduser().resolve()
    if not source_path.is_dir():
        raise FileNotFoundError(f"Data-secured source is not a directory: {source_path}")

    preferred: list[str] = []
    fallback: list[str] = []
    for path in source_path.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source_path).as_posix()
        if any(part.startswith(".") for part in Path(relative).parts):
            fallback.append(relative)
        else:
            preferred.append(relative)
            if len(preferred) >= limit:
                break

    sample = preferred[:limit]
    if len(sample) < limit:
        sample.extend(fallback[: limit - len(sample)])
    return sample


def build_file_server_sync_command(
    source: Path | str,
    destination: Path | str,
    *,
    dry_run: bool = False,
    files_from: Path | str | None = None,
) -> list[str]:
    cmd = ["rsync", "-a", "--update", "--human-readable", "--itemize-changes"]
    if dry_run:
        cmd.append("--dry-run")
    if files_from is not None:
        cmd.extend(["--files-from", str(files_from)])
    cmd.extend([f"{Path(source).expanduser().resolve()}/", f"{Path(destination).expanduser().resolve()}/"])
    return cmd


def build_box_sync_command(
    source: Path | str,
    *,
    remote: str,
    folder: str,
    dry_run: bool = False,
    files_from: Path | str | None = None,
) -> list[str]:
    target = f"{remote}:{folder}" if folder else f"{remote}:"
    cmd = ["rclone", "copy"]
    if dry_run:
        cmd.append("--dry-run")
    if files_from is not None:
        cmd.extend(["--files-from", str(files_from)])
    cmd.extend(
        [
            str(Path(source).expanduser().resolve()),
            target,
            "--update",
            "--create-empty-src-dirs",
            "--copy-links",
            "-v",
        ]
    )
    return cmd


def _write_files_from(sample_files: Sequence[str]) -> tempfile.TemporaryDirectory[str] | None:
    if not sample_files:
        return None
    tmpdir = tempfile.TemporaryDirectory(prefix="data_secured_sync_")
    files_from = Path(tmpdir.name) / "files_from.txt"
    files_from.write_text("\n".join(sample_files) + "\n", encoding="utf-8")
    return tmpdir


def _run_command(label: str, cmd: Sequence[str]) -> bool:
    logger.info("%s sync command: %s", label, " ".join(cmd))
    result = subprocess.run(list(cmd), check=False)
    if result.returncode == 0:
        logger.info("%s sync completed.", label)
        return True
    logger.error("%s sync failed with exit code %s.", label, result.returncode)
    return False


def sync_data_secured(
    raw_cfg: Mapping[str, Any],
    *,
    dry_run: bool = False,
    sample_files: Sequence[str] | None = None,
    destinations: Sequence[str] | None = None,
) -> dict[str, bool]:
    """Incrementally copy Data-secured to the file server and Box.

    This intentionally uses copy semantics rather than sync semantics so files
    missing from the destinations are not deleted.
    """

    cfg = load_data_secured_sync_config(raw_cfg)
    if not cfg.enabled:
        logger.info("Data-secured sync disabled; skipping.")
        return {}
    if not cfg.source.is_dir():
        raise FileNotFoundError(f"Data-secured source is not a directory: {cfg.source}")

    selected = set(destinations or ("file_server", "box"))
    tmpdir = _write_files_from(sample_files or [])
    files_from = Path(tmpdir.name) / "files_from.txt" if tmpdir is not None else None

    if sample_files:
        logger.info("Running Data-secured verification sync with %d file(s).", len(sample_files))

    results: dict[str, bool] = {}
    try:
        if "file_server" in selected and cfg.file_server.enabled:
            cfg.file_server.path.mkdir(parents=True, exist_ok=True)
            command = build_file_server_sync_command(
                cfg.source,
                cfg.file_server.path,
                dry_run=dry_run,
                files_from=files_from,
            )
            results["file_server"] = _run_command("File server", command)

        if "box" in selected and cfg.box.enabled:
            command = build_box_sync_command(
                cfg.source,
                remote=cfg.box.remote,
                folder=cfg.box.folder,
                dry_run=dry_run,
                files_from=files_from,
            )
            results["box"] = _run_command("Box", command)
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()

    if not results:
        logger.info("No Data-secured destinations enabled for the requested sync.")
    return results
