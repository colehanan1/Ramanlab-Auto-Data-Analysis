"""python -m rigmon --config config/rigmon.yaml"""

from __future__ import annotations

import argparse
import logging
import socket
from pathlib import Path

import uvicorn

from .app import create_app
from .config import REPO, load_config
from .finalize import Finalizer
from .hub import Hub
from .notify import Notifier
from .store import Store
from .transfer import TransferWorkers, TrialTransfer


def bind_socket(host: str, port: int) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(128)
    s.set_inheritable(True)
    return s


def build_mountcheck(cfg, store, notifier):
    """(runner, None) when ready, (None, None) when disabled, (None, reason) when it
    cannot start: the hub must still come up without the mount check."""
    log = logging.getLogger("rigmon")
    if not cfg.mount_check.enabled:
        return None, None
    if not cfg.pipeline_config.exists():
        reason = (f"pipeline_config not found: {cfg.pipeline_config} "
                  "(set pipeline_config in rigmon.yaml or disable mount_check)")
        log.error("mount check disabled: %s", reason)
        return None, reason
    try:
        from fbpipe.config import load_settings

        from .mountcheck.runner import MountCheckRunner

        return MountCheckRunner(cfg, store, notifier, load_settings(cfg.pipeline_config)), None
    except Exception as e:
        reason = f"could not load {cfg.pipeline_config}: {e}"[:300]
        log.exception("mount check disabled")
        return None, reason


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="rigmon")
    ap.add_argument("--config", type=Path, default=REPO / "config" / "rigmon.yaml")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per rig per second otherwise

    cfg = load_config(args.config)
    store = Store(cfg.db_path)
    n = store.fail_interrupted_sessions()
    if n:
        logging.getLogger("rigmon").warning("marked %d interrupted finalize(s) failed", n)
    m = store.fail_interrupted_mount_checks()
    if m:
        logging.getLogger("rigmon").warning("marked %d interrupted mount check(s) failed", m)
    notifier = Notifier(cfg, store)
    transfer = TrialTransfer(cfg, store)
    finalizer = Finalizer(cfg, store, transfer, alert=notifier.alert)
    mountcheck, mc_error = build_mountcheck(cfg, store, notifier)
    hub = Hub(cfg, store, notifier, transfer, finalizer, TransferWorkers(), mountcheck=mountcheck,
               mount_check_error=mc_error)
    transfer.is_recording = hub.is_recording   # never read a Pi while it is recording
    server = uvicorn.Server(uvicorn.Config(create_app(hub), log_level="warning"))
    server.run(sockets=[bind_socket(h, cfg.port) for h in cfg.bind])


if __name__ == "__main__":
    main()
