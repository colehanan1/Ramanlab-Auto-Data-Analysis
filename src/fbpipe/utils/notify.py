"""Lightweight ntfy.sh push notifications.

Mirrors the helper in ``scripts/pipeline/run_workflows.py`` but importable from
anywhere in the package. Topic defaults to ``ramanlab-pipeline`` and can be
overridden with the ``NTFY_TOPIC`` env var. Set ``FBPIPE_DISABLE_NTFY=1`` to
suppress all notifications (useful for tests / offline runs).
"""

from __future__ import annotations

import logging
import os
import urllib.request

log = logging.getLogger("fbpipe.notify")

DEFAULT_NTFY_TOPIC = "ramanlab-pipeline"


def ntfy_notify(
    title: str,
    message: str,
    priority: str = "default",
    tags: str = "",
    topic: str | None = None,
) -> bool:
    """POST a notification to ntfy.sh. Returns True on success, never raises."""

    if os.getenv("FBPIPE_DISABLE_NTFY", "").strip().lower() in {"1", "true", "yes", "on"}:
        log.info("[ntfy disabled] %s: %s", title, message)
        return False

    topic = topic or os.getenv("NTFY_TOPIC", DEFAULT_NTFY_TOPIC)
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{topic}", data=message.encode("utf-8"), method="POST"
        )
        req.add_header("Title", title)
        req.add_header("Priority", priority)
        if tags:
            req.add_header("Tags", tags)
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception as exc:  # pragma: no cover - network dependent
        log.warning("ntfy notify failed: %s", exc)
        return False


__all__ = ["ntfy_notify", "DEFAULT_NTFY_TOPIC"]
