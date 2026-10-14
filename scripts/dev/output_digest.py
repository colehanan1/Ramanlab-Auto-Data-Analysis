#!/usr/bin/env python3
"""Content digests of output trees, for the PER parity check.

Before the nightly is switched from ``config/config_new.yaml`` to the layered ``config/config_new_abdomen.yaml`` the PER
outputs have to be shown unchanged. Snapshot the output trees after a ``config_new`` run (A), run ``config_new`` again
and snapshot (A'), run the new config and snapshot (B); then compare B with A while ignoring every path that already
differs between A and A' -- whatever is not reproducible from one run to the next (SVG ids, PDF dates, ...)::

    python3 scripts/dev/output_digest.py snapshot --root DIR [--root DIR ...] [--exclude GLOB ...] [--workers N] \\
        --out A.json
    python3 scripts/dev/output_digest.py compare A.json B.json [--noise A2.json]      # exit 0 = unchanged, 1 = differs

``--exclude 'Abdomen-Geometry*' --exclude 'Mounting-Posture*' --exclude 'Abdomen-Movement*'`` leaves out the abdomen and
mounting stages' own trees. Only reads the roots; ``--out`` must lie outside them (a snapshot file inside a root would
show up as a new file in the next one; symlinks are resolved for that check). ``snapshot`` never overwrites an existing
``--out`` unless ``--force``: A.json is the irreplaceable reference.

``compare`` exits 0 (unchanged), 1 (differs) or 2 (it refused, or a file was unusable). It refuses snapshots that cannot
honestly be compared, because each would end in a false "identical": any empty snapshot, snapshots taken from other
roots or with other ``--exclude`` globs (a wrong noise snapshot turns every path missing from it into "noise"; a trailing
slash is no part of a glob, so ``x*/`` and ``x*`` count as the same one), and the same file given twice. And it prints every path it ignored as noise, with counts per extension, svg and pdf last: those
are rewritten with fresh ids and dates on every run, so what matters is whether a PNG, CSV or parquet is on the list.

A digest is a function of what a file CONTAINS, not of how it was written:

* ``.csv`` and every other file -- sha256 of the raw bytes, streamed. A CSV is never parsed (the wide one is 1.4 GB);
* ``.parquet`` -- sha256 of the Arrow IPC bytes of the schema (names, types, nullability; schema and field metadata
  stripped) and of the data in batches of a fixed row count taken from that schema. Write metadata, compression and
  row-group layout do not matter; dictionary columns are decoded first, because IPC batches carry codes only;
* ``.png`` -- sha256 of the decoded pixels plus mode and size; text chunks (Software, dates, ...) and compression do
  not matter, and a palette image is compared by the colours it draws;
* ``.json`` -- sha256 of ``json.dumps(obj, sort_keys=True, separators=(",", ":"))``; one over 64 MiB is hashed as raw
  bytes instead of being parsed into memory.

Digest keys are ``<absolute root>::<path relative to the root>``, so several roots share one snapshot. A file whose
content cannot be parsed (a corrupt parquet, PNG or JSON) gets ``error:<ExceptionType>:<sha256 of its raw bytes>``
rather than crashing the snapshot: the same corrupt file still compares equal and a different one compares as changed.
A file whose raw bytes cannot be read either (permission, I/O error, a dangling link, a FIFO) gets the bare
``error:<ExceptionType>``, and any two of those compare equal. ``snapshot`` and ``compare`` count every ``error:``
digest out loud, because an unparsed file has not had its content compared. Two other entries are not files: a
directory that could not be listed (key ending in ``/``) is the bare ``error:<ExceptionType>``, and a symlinked
directory is ``symlink:<target>`` and is not followed. A symlink to a file is hashed by the content it points at.

``exclude`` globs (fnmatch, case-sensitive) match any single path component or the whole relative path, so
``Abdomen-Geometry*`` drops those directories at any depth; a directory that matches is not entered.

Files are digested by a few threads (hashlib, Pillow and pyarrow all release the GIL). Memory is bounded by the worker
count times one decoded image, and by two parquet files at a time (the real 15,000-column single-row-group tables cost
~2 GB each while they are read).
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import stat
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Mapping, Optional, Sequence

# Bump with anything below that changes what a digest means. 1: first version. 2: ``error:<ExceptionType>`` became
# ``error:<ExceptionType>:<sha256 of the raw bytes>`` for a file that can be read but not parsed.
SNAPSHOT_VERSION = 2
KEY_SEP = "::"
LISTED = 50                   # paths ``compare`` prints per category
_VOLATILE_SUFFIXES = (".svg", ".pdf")      # carry random ids and dates: they differ on every run
DEFAULT_WORKERS = 8

_READ_CHUNK = 1 << 20         # raw bytes are hashed 1 MiB at a time
_JSON_PARSE_LIMIT = 64 << 20  # bigger JSON is hashed raw instead of parsed
# Parquet is hashed in batches of ``_batch_rows(schema)`` rows. That count comes from the logical schema alone (never
# from the file's row groups or encodings), so equal content is batched alike however it was written. These four
# constants therefore belong to the digest: change one and every parquet digest changes with it.
_PARQUET_BATCH_BYTES = 128 << 20
_MIN_BATCH_ROWS, _MAX_BATCH_ROWS = 64, 1 << 20
_VARIABLE_WIDTH_BYTES = 32    # assumed size of a string / binary / nested value when sizing a batch
_PARQUET_SLOTS = threading.BoundedSemaphore(2)


# --------------------------------------------------------------------------
# One file
# --------------------------------------------------------------------------

def _hash_bytes(path: Path) -> str:
    h = hashlib.sha256()
    buffer = bytearray(_READ_CHUNK)
    view = memoryview(buffer)
    with open(path, "rb", buffering=0) as f:
        while count := f.readinto(buffer):
            h.update(view[:count])
    return h.hexdigest()


def _hash_json(path: Path) -> str:
    if os.stat(path).st_size > _JSON_PARSE_LIMIT:
        return _hash_bytes(path)
    with open(path, "rb") as f:
        obj = json.loads(f.read())
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _hash_png(path: Path) -> str:
    from PIL import Image

    with Image.open(path) as image:
        drawn = image.convert("RGBA") if image.mode in ("P", "PA") else image   # palette indices are not pixels
        h = hashlib.sha256()
        h.update(f"{drawn.mode}|{drawn.width}x{drawn.height}|".encode("ascii"))
        h.update(drawn.tobytes())
    return h.hexdigest()


def _value_bytes(arrow_type) -> int:
    try:
        return max(1, arrow_type.bit_width // 8)
    except ValueError:                             # string, binary, list, struct: no fixed width
        return _VARIABLE_WIDTH_BYTES


def _batch_rows(schema) -> int:
    width = sum(_value_bytes(field.type) for field in schema) or 1
    return max(_MIN_BATCH_ROWS, min(_MAX_BATCH_ROWS, _PARQUET_BATCH_BYTES // width))


def _hash_parquet(path: Path) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq

    with _PARQUET_SLOTS:
        parquet = pq.ParquetFile(path)
        try:
            # Rebuilt field by field: that drops the schema-level and field-level metadata, which hold the writer and
            # the write time. A dictionary column becomes its value type, because a serialised batch carries only the
            # codes of a dictionary column and two different dictionaries would hash alike.
            schema = pa.schema([pa.field(f.name, f.type.value_type if pa.types.is_dictionary(f.type) else f.type,
                                         f.nullable) for f in parquet.schema_arrow])
            h = hashlib.sha256()
            h.update(schema.serialize())
            for batch in parquet.iter_batches(batch_size=_batch_rows(schema)):
                columns = [c.dictionary_decode() if pa.types.is_dictionary(c.type) else c for c in batch.columns]
                h.update(pa.RecordBatch.from_arrays(columns, schema=schema).serialize())
        finally:
            parquet.close()
    return h.hexdigest()


_HASHERS = {".parquet": _hash_parquet, ".png": _hash_png, ".json": _hash_json}


def _with_raw_hash(path: Path, error: str) -> str:
    """``error`` plus the sha256 of the file's raw bytes, so that a different corrupt file still compares as changed."""
    try:
        return f"{error}:{_hash_bytes(path)}"
    except Exception:                              # noqa: BLE001 -- not even readable: the bare error is all there is
        return error


def digest_file(path) -> str:
    """The digest of one file (see the module docstring), or an ``error:`` digest if it cannot be read or parsed."""
    path = Path(path)
    try:
        if not stat.S_ISREG(os.stat(path).st_mode):
            return "error:NotARegularFile"         # opening a FIFO or a device for reading can block for ever
        return _HASHERS.get(path.suffix.lower(), _hash_bytes)(path)
    except ImportError:
        raise                                      # a missing library is not a bad file: every digest would be wrong
    except Exception as exc:                       # noqa: BLE001 -- a bad file must never abort a snapshot
        return _with_raw_hash(path, f"error:{type(exc).__name__}")


# --------------------------------------------------------------------------
# A tree
# --------------------------------------------------------------------------

def _excluded(parts: Sequence[str], patterns: Sequence[str]) -> bool:
    if not patterns:
        return False
    whole = "/".join(parts)
    return any(fnmatch.fnmatchcase(whole, p) or any(fnmatch.fnmatchcase(part, p) for part in parts) for p in patterns)


def _size(path: str) -> int:
    try:
        return os.stat(path).st_size
    except OSError:
        return 0


def _link_digest(path: str) -> str:
    try:
        return f"symlink:{os.readlink(path)}"
    except OSError as exc:
        return f"error:{type(exc).__name__}"


def _digest_files(jobs: Sequence[tuple], workers: Optional[int]) -> dict:
    jobs = sorted(jobs, key=lambda job: job[2], reverse=True)       # start the long ones first
    if workers is None:
        workers = min(DEFAULT_WORKERS, os.cpu_count() or 1)
    if workers <= 1 or len(jobs) <= 1:
        return {key: digest_file(path) for key, path, _ in jobs}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        try:
            results = list(pool.map(digest_file, [path for _, path, _ in jobs]))
        except BaseException:                                       # Ctrl-C: do not first work through the queue
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    return {key: result for (key, _, _), result in zip(jobs, results)}


def digest_tree(root, *, exclude: Sequence[str] = (), workers: Optional[int] = None) -> dict:
    """``{"<absolute root>::<relative path>": digest}`` for every file under ``root``, sorted by key.

    ``exclude`` are fnmatch globs matched against each path component and against the whole relative path.
    ``workers`` is the thread count (default 8, never more than the CPU count; 1 = serial).
    """
    base = os.path.abspath(os.fspath(root))
    prefix = base + KEY_SEP
    patterns = [p.rstrip("/") for p in exclude]
    digests: dict = {}
    jobs: list = []

    def listing_failed(exc: OSError) -> None:
        if os.path.abspath(exc.filename) == base:  # a missing, non-directory or unreadable root: an error, not an
            raise exc                              # empty snapshot
        rel = os.path.relpath(exc.filename, base).replace(os.sep, "/")
        digests[f"{prefix}{rel}/"] = f"error:{type(exc).__name__}"

    for dirpath, dirnames, filenames in os.walk(base, onerror=listing_failed):
        here = os.path.relpath(dirpath, base)
        parts = () if here == "." else tuple(here.split(os.sep))
        entered = []
        for name in dirnames:
            if _excluded(parts + (name,), patterns):
                continue
            full = os.path.join(dirpath, name)
            if os.path.islink(full):
                digests[prefix + "/".join(parts + (name,))] = _link_digest(full)
            else:
                entered.append(name)
        dirnames[:] = entered
        for name in filenames:
            if not _excluded(parts + (name,), patterns):
                full = os.path.join(dirpath, name)
                jobs.append((prefix + "/".join(parts + (name,)), full, _size(full)))

    digests.update(_digest_files(jobs, workers))
    return dict(sorted(digests.items()))


# --------------------------------------------------------------------------
# Comparing snapshots
# --------------------------------------------------------------------------

def diff_digests(a: Mapping[str, str], b: Mapping[str, str]) -> dict:
    """What turns ``a`` into ``b``: keys only in ``b`` (added), only in ``a`` (removed), in both with another digest."""
    return {
        "added": sorted(b.keys() - a.keys()),
        "removed": sorted(a.keys() - b.keys()),
        "changed": sorted(k for k in a.keys() & b.keys() if a[k] != b[k]),
    }


def compare(a: Mapping[str, str], b: Mapping[str, str], noise: Optional[Mapping[str, str]] = None) -> dict:
    """``diff_digests(a, b)`` without every path that differs in any way between ``a`` and ``noise``.

    Besides ``added`` / ``removed`` / ``changed`` it reports ``noise`` (every path ``a`` and ``noise`` disagree on) and
    ``dropped`` (the part of it that also differs between ``a`` and ``b``, and so was left out of the three lists).
    """
    result = diff_digests(a, b)
    noisy = set()
    if noise is not None:
        noisy = {path for paths in diff_digests(a, noise).values() for path in paths}
    differing = {path for paths in result.values() for path in paths}
    report = {kind: [path for path in paths if path not in noisy] for kind, paths in result.items()}
    report["noise"] = sorted(noisy)
    report["dropped"] = sorted(noisy & differing)
    return report


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def _load_snapshot(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        snapshot = json.load(f)
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("digests"), dict):
        raise ValueError(f"{path}: not an output-digest snapshot")
    if snapshot.get("version") != SNAPSHOT_VERSION:
        raise ValueError(f"{path}: snapshot version {snapshot.get('version')!r}, but this tool reads version "
                         f"{SNAPSHOT_VERSION}: take the snapshot again")
    for field in ("roots", "exclude"):
        value = snapshot.get(field)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"{path}: not an output-digest snapshot (no usable {field!r} list)")
    if not all(isinstance(digest, str) for digest in snapshot["digests"].values()):
        raise ValueError(f"{path}: not an output-digest snapshot (a digest is not a string)")
    return snapshot


def _require_comparable(named: Sequence[tuple]) -> None:
    """Refuse snapshots that cannot honestly be compared; ``named`` is ``[(role, file, snapshot), ...]``, A first.

    A noise snapshot of other paths than A's turns every path missing from it into "noise", and the same file given
    twice turns every difference into one: both end in a false "identical". An empty snapshot proves nothing.
    """
    who = [f"{role} snapshot {name}" for role, name, _ in named]
    files = [os.path.realpath(name) for _, name, _ in named]
    for i in range(1, len(named)):
        if files[i] in files[:i]:
            raise ValueError(f"{who[i]} is the same file as {who[files.index(files[i])]}: "
                             "compare needs distinct snapshots")
    for label, (_, _, snapshot) in zip(who, named):
        if not snapshot["digests"]:
            raise ValueError(f"{label}: holds no files (a wrong --root, or everything excluded?)")
    reference = named[0][2]
    for label, (_, _, snapshot) in zip(who[1:], named[1:]):
        for field, what in (("roots", "root(s)"), ("exclude", "--exclude glob(s)")):
            # A trailing slash is no part of a glob (``digest_tree`` strips it), so ``x*/`` and ``x*`` are one exclude:
            # refusing the pair would cost a re-run of the snapshot, over a typo, for nothing.
            theirs, ours = (sorted({g.rstrip("/") if field == "exclude" else g for g in values})
                            for values in (snapshot[field], reference[field]))
            if theirs != ours:
                raise ValueError(f"{label}: taken with other {what} than {who[0]}: {theirs} against {ours}; "
                                 "snapshots of different trees cannot be compared")


def _print_paths(label: str, paths: Sequence[str]) -> None:
    print(f"{label}: {len(paths)}")
    for path in paths[:LISTED]:
        print(f"  {path}")
    if len(paths) > LISTED:
        print(f"  ... and {len(paths) - LISTED} more")


def _suffix(path: str) -> str:
    return os.path.splitext(path)[1].lower() or "(none)"


def _print_ignored_noise(dropped: Sequence[str]) -> None:
    """Every path ``compare`` ignored as noise (capped), svg and pdf last, with the counts per extension."""
    print(f"ignored as noise: {len(dropped)}")
    if not dropped:
        return
    counts = Counter(_suffix(path) for path in dropped)
    by_extension = sorted(counts.items(), key=lambda item: (item[0] in _VOLATILE_SUFFIXES, item[0]))
    print("  by extension (svg and pdf last): " + ", ".join(f"{extension} {n}" for extension, n in by_extension))
    ordered = sorted(dropped, key=lambda path: (_suffix(path) in _VOLATILE_SUFFIXES, path))
    for path in ordered[:LISTED]:
        print(f"  {path}")
    if len(ordered) > LISTED:
        print(f"  ... and {len(ordered) - LISTED} more")


def _error_kind(digest: str) -> str:
    """``error:<ExceptionType>`` of an error digest, whatever raw-byte hash follows it."""
    return ":".join(digest.split(":", 2)[:2])


def _has_raw_hash(digest: str) -> bool:
    return digest.count(":") == 2


def _unreadable_line(path: str, digest: str) -> str:
    how = "raw bytes identical" if _has_raw_hash(digest) else "NOT compared"
    return f"{path}  [{_error_kind(digest)}; {how}]"


def _inside(path: str, root: str) -> bool:
    return os.path.commonpath([path, root]) == root


def _cmd_snapshot(args: argparse.Namespace) -> int:
    roots = [os.path.abspath(r) for r in args.root]
    out = os.path.abspath(args.out)
    for root in roots:
        if not os.path.isdir(root):
            raise NotADirectoryError(f"--root is not a directory: {root}")
        if _inside(os.path.realpath(out), os.path.realpath(root)):
            raise ValueError(f"--out {out} lies inside the root {root} (symlinks resolved): "
                             "the next snapshot would digest it")
    if os.path.isdir(out):
        raise IsADirectoryError(f"--out is a directory: {out}")
    if os.path.lexists(out) and not args.force:
        raise FileExistsError(f"--out already exists: {out} (a reference snapshot is not re-taken; "
                              "pass --force to overwrite it)")
    os.makedirs(os.path.dirname(out), exist_ok=True)         # a bad --out should fail now, not after the digesting
    exclude = args.exclude or []
    started = time.monotonic()
    digests: dict = {}
    for root in roots:
        print(f"[digest] {root} ...", file=sys.stderr, flush=True)
        digests.update(digest_tree(root, exclude=exclude, workers=args.workers))
    payload = {"version": SNAPSHOT_VERSION, "roots": roots, "exclude": list(exclude),
               "digests": dict(sorted(digests.items()))}
    mode = "w" if args.force else "x"                        # "x": never over a file that appeared meanwhile
    with open(out, mode, encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
        f.write("\n")
    errors = Counter(_error_kind(d) for d in digests.values() if d.startswith("error:"))
    unreadable = ""
    if errors:
        kinds = ", ".join(f"{kind} x{n}" for kind, n in sorted(errors.items()))
        unreadable = f", {sum(errors.values())} unreadable ({kinds})"
    print(f"{len(digests)} files digested from {len(roots)} root(s) in {time.monotonic() - started:.1f}s"
          f"{unreadable} -> {out}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    named = [("A", args.a, _load_snapshot(args.a)), ("B", args.b, _load_snapshot(args.b))]
    if args.noise is not None:
        named.append(("noise", args.noise, _load_snapshot(args.noise)))
    _require_comparable(named)
    a, b = named[0][2], named[1][2]
    noise = named[2][2] if args.noise is not None else None
    result = compare(a["digests"], b["digests"], noise["digests"] if noise else None)
    print(f"A: {len(a['digests'])} files ({args.a})")
    print(f"B: {len(b['digests'])} files ({args.b})")
    if noise is not None:
        print(f"noise: {len(result['noise'])} paths differ between A and {args.noise}")
    for kind in ("changed", "added", "removed"):
        _print_paths(kind, result[kind])
    if noise is not None:
        _print_ignored_noise(result["dropped"])
    unreadable = sorted(p for p, d in a["digests"].items() if d.startswith("error:") and b["digests"].get(p) == d)
    if unreadable:
        _print_paths("WARNING: unreadable in both snapshots",
                     [_unreadable_line(p, a["digests"][p]) for p in unreadable])
    remaining = sum(len(result[kind]) for kind in ("added", "removed", "changed"))
    scope = " (noise removed)" if noise is not None else ""
    verdict = f"RESULT: {remaining} difference(s){scope}" if remaining else f"RESULT: identical{scope}"
    if unreadable:
        blind = sum(1 for p in unreadable if not _has_raw_hash(a["digests"][p]))
        verdict += f"; {len(unreadable)} unreadable path(s) in both snapshots"
        verdict += f", {blind} of them NOT compared" if blind else ""
    print(verdict)
    return 1 if remaining else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Content digests of output trees, for the PER parity check.")
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser("snapshot", help="digest every file under one or more roots into a JSON file")
    snapshot.add_argument("--root", action="append", required=True, metavar="DIR", help="tree to digest (repeatable)")
    snapshot.add_argument("--exclude", action="append", metavar="GLOB",
                          help="skip paths with a component, or a whole relative path, matching GLOB (repeatable)")
    snapshot.add_argument("--workers", type=int, metavar="N", help=f"threads (default {DEFAULT_WORKERS}; 1 = serial)")
    snapshot.add_argument("--out", required=True, metavar="X.json", help="snapshot file to write (outside every root)")
    snapshot.add_argument("--force", action="store_true", help="overwrite an existing --out file (refused otherwise)")
    snapshot.set_defaults(run=_cmd_snapshot)
    compare_ = commands.add_parser("compare", help="B against A, minus what already differs between A and --noise")
    compare_.add_argument("a", metavar="A.json", help="reference snapshot")
    compare_.add_argument("b", metavar="B.json", help="snapshot to check against it")
    compare_.add_argument("--noise", metavar="A2.json",
                          help="a second snapshot of A's own code: paths that differ from A are ignored")
    compare_.set_defaults(run=_cmd_compare)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.run(args)
    except (OSError, ValueError) as exc:           # a bad path or snapshot file: a message, not a traceback
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
