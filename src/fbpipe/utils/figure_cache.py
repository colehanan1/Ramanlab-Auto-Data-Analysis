"""Skip redrawing a figure whose inputs did not change.

A run rewrites the wide tables whenever any dataset gains a fly, so file times
cannot say which figures are stale. Each figure (or set of figures) is keyed by a
digest of the rows it is drawn from and of the code that draws it; delete the cache
file to force a full redraw.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
_OWN_PACKAGES = ("scripts", "fbpipe")


def _module_files(root: Path, dotted: str) -> list[Path]:
    """The file behind a dotted module name, with the package __init__s above it."""
    parts = dotted.split(".")
    base = root / "src" if parts[0] == "fbpipe" else root
    found = []
    for i in range(1, len(parts) + 1):
        here = base.joinpath(*parts[:i])
        if (here / "__init__.py").is_file():
            found.append(here / "__init__.py")
        elif i == len(parts) and here.with_suffix(".py").is_file():
            found.append(here.with_suffix(".py"))
    return found


def _is_python(node: ast.AST) -> bool:
    """Whether a command's first element is a Python interpreter: PY, PYTHON or any name
    with "python" in it ("python3" too), sys.executable, or a call wrapping one, such
    as ``str(Path(python_exec))``."""
    for n in ast.walk(node):
        if (isinstance(n, ast.Attribute) and n.attr == "executable"
                and isinstance(n.value, ast.Name) and n.value.id == "sys"):
            return True
        name = (n.id if isinstance(n, ast.Name) else n.attr if isinstance(n, ast.Attribute)
                else n.value if isinstance(n, ast.Constant) and isinstance(n.value, str)
                else "")
        if name.upper() == "PY" or ("python" in name.lower() and not name.endswith(".py")):
            return True
    return False


def _spelled(node: ast.AST) -> str | None:
    """The text a string spells, or a ``ROOT / "scripts" / "x.py"`` chain from its first
    string on; None for anything else."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        head, tail = _spelled(node.left), _spelled(node.right)
        if tail is not None:
            return tail if head is None else f"{head}/{tail}"
    return None


def _script_names(node: ast.AST) -> Iterator[str]:
    """Each ``.py`` path spelled anywhere inside one element of a command."""
    spelled = _spelled(node)
    if spelled is None:
        for child in ast.iter_child_nodes(node):
            yield from _script_names(child)
    elif spelled.endswith(".py"):
        yield spelled


def _launched(tree: ast.AST, root: Path) -> list[Path]:
    """Scripts a module runs: ``.py`` strings inside a list or tuple led by a Python
    interpreter, as in ``[PY, a + "x.py", ...]``. A ``.py`` string anywhere else runs
    nothing (2026-11-09: autofreeze's tuple of provenance globs pulled
    envelope_combined.py and its imports into the closure). A bare ``x.py`` is in
    scripts/analysis; a path with a folder is read from the repo root."""
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, (ast.List, ast.Tuple)) and node.elts
                and _is_python(node.elts[0])):
            continue
        for name in (n for element in node.elts[1:] for n in _script_names(element)):
            script = root / name if "/" in name else root / "scripts" / "analysis" / name
            if script.is_file() and script.resolve().is_relative_to(root):
                found.append(script)
    return found


def _dependencies(path: Path, root: Path) -> list[Path]:
    """Repo files ``path`` imports (absolute or relative) or launches in a command."""
    text = path.read_text(errors="replace")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    inside = path.relative_to(root / "src" if path.is_relative_to(root / "src") else root)
    package = list(inside.with_suffix("").parts[:-1])
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                if node.level - 1 > len(package):
                    continue
                base = package[: len(package) - (node.level - 1)]
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            # "from pkg import name" may name a submodule as well as an attribute
            names = [module] + [f"{module}.{a.name}" for a in node.names]
        else:
            continue
        for name in names:
            if name.split(".")[0] in _OWN_PACKAGES:
                found += _module_files(root, name)
    return found + _launched(tree, root)


def code_files(entries: Iterable[Path], root: Path = REPO_ROOT) -> list[Path]:
    """The entry scripts and every repo file they import or launch, transitively."""
    root = Path(root).resolve()
    todo, seen = [Path(e).resolve() for e in entries], set()
    while todo:
        path = todo.pop()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        todo += [d.resolve() for d in _dependencies(path, root)]
    return sorted(seen)


def code_digest(entries: Iterable[Path], root: Path = REPO_ROOT) -> str:
    """Digest of the entry scripts and every repo file they import or launch, transitively.

    Only that code counts: another script being added or edited nearby must not
    make a figure look stale (2026-11-09: three new scripts by another session
    made every cohort redraw)."""
    root = Path(root).resolve()
    h = hashlib.sha256()
    for path in code_files(entries, root):
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def file_digest(path: Path | str) -> str:
    """Digest of a file's content, ``""`` when it does not exist."""
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def _kind(dtype) -> str:
    """How frame_digest reads a column: "number" as float64, "time" as ISO text, else "text"."""
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "time"
    if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_complex_dtype(dtype):
        return "number"
    return "text"


def _texts(column: pd.Series, kind: str) -> list[str | None]:
    """A column's values as text, None for every missing one (None, NaN, NA or NaT)."""
    missing = column.isna().to_numpy()
    return [None if m else v.isoformat() if kind == "time" else str(v)
            for v, m in zip(column, missing)]


def frame_digest(df: pd.DataFrame, sort_by: Sequence[str] = ()) -> str:
    """Digest of a table's columns and values, whatever order its rows came in and
    whichever dtypes pandas gave them.

    pandas 2.3.3 and 3.0.2 type one table differently (text object or ``str``,
    datetimes [ns] or [us]), so numbers (int, float, bool, nullable) count as
    float64, datetimes as ISO text and the rest as ``str``; None, NaN, NA and NaT
    are one missing value. Rows go in the order of their own digests, so ``sort_by``
    (kept for the callers) is no longer needed and rows that share a key cannot
    reorder the digest either.
    """
    kinds = [_kind(dtype) for dtype in df.dtypes]
    numbers = df.iloc[:, [i for i, k in enumerate(kinds) if k == "number"]]
    values = np.array(numbers.to_numpy(dtype="float64", na_value=np.nan), dtype="<f8",
                      order="C")
    values += 0.0  # -0.0 is 0.0
    values[np.isnan(values)] = np.nan  # one NaN, whatever its bits
    texts = [_texts(df.iloc[:, i], k) for i, k in enumerate(kinds) if k != "number"]
    rows = []
    for numeric, text in zip(values, zip(*texts) if texts else [()] * len(df)):
        h = hashlib.sha256(numeric)
        h.update(repr(text).encode())
        rows.append(h.digest())
    h = hashlib.sha256(repr([(str(c), k) for c, k in zip(df.columns, kinds)]).encode())
    h.update(b"".join(sorted(rows)))
    return h.hexdigest()


def combine(*parts: object) -> str:
    return hashlib.sha256(repr([str(p) for p in parts]).encode()).hexdigest()


class FigureCache:
    """``{key: fingerprint}`` kept in one JSON file; a missing or corrupt file is empty."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        try:
            self._entries = dict(json.loads(self.path.read_text()))
        except (OSError, ValueError, TypeError):
            self._entries = {}

    def is_current(self, key: str, fingerprint: str) -> bool:
        return self._entries.get(key) == fingerprint

    def record(self, key: str, fingerprint: str) -> None:
        self._entries[key] = fingerprint

    def forget(self, key: str) -> None:
        self._entries.pop(key, None)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.tmp")
        tmp.write_text(json.dumps(self._entries, indent=0, sort_keys=True))
        os.replace(tmp, self.path)
