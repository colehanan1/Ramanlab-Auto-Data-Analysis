"""Children that die with their parent (Linux ``PR_SET_PDEATHSIG``).

The abdomen stage and the geometry fan-out start their children in process groups of their own, so that a timeout can
kill a group as one. When the PARENT is killed nothing is left to do that: a SIGTERM or SIGHUP can be caught (the driver
does, and kills the group), but a SIGKILL cannot, and the children would run on with no deadline and, in the driver's
case, with the abdomen lock freed under them.

:func:`die_with_parent_preexec` is the ``preexec_fn`` that asks the kernel to send the child SIGTERM when its parent
dies. The setting survives ``exec`` (it is cleared only for set-user-ID binaries), so a stage command that fans out
workers of its own gives them the same guard, one level down.

Linux only: elsewhere, or where libc cannot be reached, it returns None, and ``Popen(preexec_fn=None)`` is what Popen does
anyway. The signal follows the parent THREAD that started the child, so start children from a thread that outlives them
(the main thread of the driver). ``preexec_fn`` runs between fork and exec; to keep that window as short and as safe as it
can be, libc is resolved here in the parent and the child only makes one call.
"""
from __future__ import annotations

import ctypes
import os
import signal
import sys
from typing import Callable, Optional

PR_SET_PDEATHSIG = 1


def die_with_parent_preexec(signum: int = signal.SIGTERM) -> Optional[Callable[[], None]]:
    """A ``preexec_fn`` for ``subprocess.Popen`` that makes the child receive *signum* when its parent dies; None where
    that is not available. Never raises: a guard that cannot be set up is no reason to stop the command it guards."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        prctl = ctypes.CDLL(None, use_errno=True).prctl
        prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
        prctl.restype = ctypes.c_int
    except (OSError, AttributeError):
        return None
    parent = os.getpid()

    def preexec() -> None:
        try:
            prctl(PR_SET_PDEATHSIG, int(signum), 0, 0, 0)
        except Exception:                                  # noqa: BLE001  (the command must still start)
            return
        if os.getppid() != parent:
            # The parent died before the call took effect, so nothing will ever signal this child: do what the kernel
            # would have done. (os._exit, not a signal: a Python handler inherited from the parent could swallow it.)
            os._exit(128 + int(signum))

    return preexec
