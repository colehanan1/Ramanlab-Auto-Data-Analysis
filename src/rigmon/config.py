"""Load config/rigmon.yaml (gitignored) into frozen dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

from fbpipe.utils.mount_qc import Thresholds

REPO = Path(__file__).resolve().parents[2]
DEFAULT_REMUX_ARGS = ("--crf", "22", "--preset", "slow", "--faststart",
                      "--scale", "1.0", "--fps-cap", "50.0")


@dataclass(frozen=True)
class RigConfig:
    name: str              # the Pi's hostname (what it sends as run.host)
    label: str             # "Pi 1"
    ip: str                # tailnet IP; "local" = test rig on this machine
    port: int = 8008
    ssh_user: str = "ramanlab"
    control_key: str = field(default="", repr=False)

    @property
    def host(self) -> str:
        return "127.0.0.1" if self.ip == "local" else self.ip

    @property
    def status_url(self) -> str:
        return f"http://{self.host}:{self.port}/status.json"

    @property
    def live_url(self) -> str:
        return f"http://{self.host}:{self.port}/"


@dataclass(frozen=True)
class MountCheckConfig:
    enabled: bool = True
    delay_after_wait_start_s: float = 1800.0
    sample_s: float = 300.0
    sample_fps: float = 2.0
    min_remaining_s: float = 1200.0
    worker_timeout_s: float = 600.0
    conf: float = 0.25
    sharpness_half_px: int = 48
    thresholds: dict = field(default_factory=dict)  # overrides for mount_qc.Thresholds

    def th(self) -> Thresholds:
        return Thresholds(**self.thresholds)


@dataclass(frozen=True)
class TracesConfig:
    enabled: bool = True
    frame_stride: int = 2
    conf: float = 0.25
    worker_timeout_s: float = 600.0


@dataclass(frozen=True)
class HubConfig:
    rigs: tuple[RigConfig, ...]
    staging_root: Path
    data_root: Path
    db_path: Path
    ssh_key: Path = Path.home() / ".ssh" / "pi_backup"
    bind: tuple[str, ...] = ("127.0.0.1",)
    port: int = 8765
    poll_interval_s: float = 1.0
    offline_after_s: float = 60.0
    transfer_enabled: bool = True
    min_free_gb: float = 20.0
    remux_python: str = "/usr/bin/python3"
    remux_script: Path = REPO / "scripts" / "tools" / "remux_from_csv.py"
    remux_args: tuple[str, ...] = DEFAULT_REMUX_ARGS
    remux_path_env: str = "/usr/bin:/bin"
    ntfy_topic: str | None = None
    pipeline_config: Path = REPO / "config" / "config_new.yaml"
    qc_root: Path | None = None
    mount_check: MountCheckConfig = field(default_factory=MountCheckConfig)
    traces: TracesConfig = field(default_factory=TracesConfig)
    transfer_bwlimit_kbps: int = 40000
    cleanup_time: str = "00:05"
    pause_reminder_s: float = 300.0

    def rig(self, name: str) -> RigConfig:
        for r in self.rigs:
            if r.name == name:
                return r
        raise KeyError(name)


_PATH_KEYS = ("staging_root", "data_root", "db_path", "ssh_key", "remux_script",
              "pipeline_config", "qc_root")
_TUPLE_KEYS = ("bind", "remux_args")


def load_config(path) -> HubConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    rigs = tuple(RigConfig(**r) for r in raw.pop("rigs", None) or [])
    if not rigs:
        raise ValueError("rigmon config needs at least one rig")
    names = [r.name for r in rigs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate rig names in {names}")
    kw = {}
    for key in _PATH_KEYS:
        if key in raw:
            kw[key] = Path(str(raw.pop(key))).expanduser()
    for key in _TUPLE_KEYS:
        if key in raw:
            kw[key] = tuple(str(v) for v in raw.pop(key))
    if "mount_check" in raw:
        mc = MountCheckConfig(**(raw.pop("mount_check") or {}))
        mc.th()  # unknown threshold keys fail here, at load time
        kw["mount_check"] = mc
    if "traces" in raw:
        kw["traces"] = TracesConfig(**(raw.pop("traces") or {}))
    kw.update(raw)  # scalars; an unknown key is a TypeError from the dataclass
    cfg = HubConfig(rigs=rigs, **kw)
    hh, _, mm = str(cfg.cleanup_time).partition(":")
    if not (hh.isdigit() and mm.isdigit() and 0 <= int(hh) < 24 and 0 <= int(mm) < 60):
        raise ValueError(f"cleanup_time must be HH:MM, got {cfg.cleanup_time!r}")
    if cfg.pause_reminder_s <= 0:
        raise ValueError(f"pause_reminder_s must be > 0, got {cfg.pause_reminder_s!r}")
    if cfg.qc_root is None:
        cfg = replace(cfg, qc_root=cfg.data_root / "rigmon_qc")
    if {"0.0.0.0", "::"} & set(cfg.bind):
        raise ValueError("bind must list specific addresses, never all interfaces")
    if "flys_New" in cfg.staging_root.parts:
        raise ValueError("staging_root must be outside flys_New (the nightly pipeline reads it)")
    if "flys_New" in cfg.qc_root.parts:
        raise ValueError("qc_root must be outside flys_New (the nightly pipeline reads it)")
    return cfg
