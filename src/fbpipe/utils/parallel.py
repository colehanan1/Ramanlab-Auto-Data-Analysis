"""Opt-in CPU parallelism helper for embarrassingly-parallel pipeline stages.

The pipeline's per-fly / per-CSV stage loops are independent (each iteration
reads and writes a disjoint file), so they can be mapped across CPU cores. This
module centralises that decision so stages stay simple and behaviour is
identical to the serial pipeline unless ``cfg.parallel.enabled`` is set.

Determinism: results are always returned in input order (joblib preserves order
with the default loky backend), and because each work item touches disjoint
files there is no shared mutable state. Parallel output therefore matches serial
output exactly.
"""

from __future__ import annotations

import os
from typing import Callable, Iterable, List, Optional, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def resolve_n_jobs(n_jobs: int) -> int:
    """Resolve a configured ``n_jobs`` to a concrete worker count.

    ``n_jobs <= 0`` means "auto": ``max(1, cpu_count - 2)`` so the machine stays
    responsive. A positive value is used as-is (capped at the cpu count).
    """
    cpu = os.cpu_count() or 1
    if n_jobs <= 0:
        return max(1, cpu - 2)
    return max(1, min(n_jobs, cpu))


def parallel_map(
    func: Callable[[T], R],
    items: Iterable[T],
    *,
    enabled: bool = False,
    n_jobs: int = 0,
    prefer: Optional[str] = None,
) -> List[R]:
    """Map ``func`` over ``items``, in parallel when ``enabled`` (else serial).

    Falls back to a plain serial loop when parallelism is disabled, when there
    are fewer than two items, or when only one worker would be used. Results are
    returned in the same order as ``items``.

    ``prefer`` is forwarded to joblib ("processes" or "threads"); when ``None``
    joblib's default (process-based loky) is used, which is correct for the
    CPU-bound, file-writing stage bodies.
    """
    work: Sequence[T] = list(items)
    if not work:
        return []

    workers = resolve_n_jobs(n_jobs) if enabled else 1
    if not enabled or workers <= 1 or len(work) <= 1:
        return [func(item) for item in work]

    # Import lazily so the dependency is only needed when parallelism is used.
    from joblib import Parallel, delayed

    return list(
        Parallel(n_jobs=workers, prefer=prefer)(delayed(func)(item) for item in work)
    )
