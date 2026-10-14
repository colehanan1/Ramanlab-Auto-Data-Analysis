"""``scripts/dev/output_digest.py``: content digests of output trees, for the PER parity check.

The nightly may only be switched to the layered abdomen config once its PER outputs are shown unchanged: snapshot the
output trees after a ``config_new`` run (A), after a second ``config_new`` run (A'), and after the new config (B), then
compare B with A ignoring every path that already differs between A and A'. Pinned here, on ``tmp_path`` trees only:

* a digest is a function of what a file CONTAINS, never of how it was written: parquet data hashes the same under other
  schema / write metadata, compression or row-group layout; PNG pixels hash the same under other text chunks or
  compression; JSON hashes the same under any key order -- and a changed cell, pixel, value or column always changes
  it (a dictionary column is hashed by its VALUES, not only by its codes);
* a CSV is hashed as raw bytes and never parsed (the real wide CSV is 1.4 GB);
* ``exclude`` globs match any path component, or the whole relative path, at any depth;
* a corrupt file gets ``error:<ExceptionType>:<sha256 of its raw bytes>`` instead of crashing the snapshot, so the same
  corrupt file still compares equal and a different one compares as changed; a file whose raw bytes cannot be read
  either (permission, I/O error, dangling link) keeps the bare ``error:<ExceptionType>``; the CLI counts and warns
  about both;
* ``compare`` removes every path that differs between A and the noise snapshot, and the CLI exits 0 when nothing
  remains and 1 when something does.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from scripts.dev import output_digest as od

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "dev" / "output_digest.py"
PIXELS = np.random.default_rng(0).integers(0, 256, size=(16, 24, 4), dtype=np.uint8)
TREE_FILES = {"wide.csv", "notes.txt", "stats.json", "sub/fig.png", "sub/deeper/t.parquet"}


def _key(root, rel: str) -> str:
    return f"{os.path.abspath(root)}::{rel}"


def _rel(digests: dict) -> dict:
    """The digests with the root prefix stripped: what two roots holding the same content have in common."""
    return {k.split("::", 1)[1]: v for k, v in digests.items()}


def _png(path: Path, pixels=PIXELS, **save_kwargs) -> Path:
    Image.fromarray(np.asarray(pixels, dtype=np.uint8)).save(path, **save_kwargs)
    return path


def _text_chunks(**chunks: str) -> PngInfo:
    info = PngInfo()
    for name, text in chunks.items():
        info.add_text(name, text)
    return info


def _tree(root: Path) -> Path:
    """A small tree with one file of every kind the digest treats differently."""
    (root / "sub" / "deeper").mkdir(parents=True)
    (root / "wide.csv").write_text("fly,t0,t1\nA,0.0,1.5\nB,2.5,3.0\n")
    (root / "notes.txt").write_text("hello\n")
    (root / "stats.json").write_text(json.dumps({"b": 1, "a": [1, 2]}))
    _png(root / "sub" / "fig.png")
    pq.write_table(pa.table({"x": [1.0, 2.0, 3.0], "s": ["a", "b", "c"]}), root / "sub" / "deeper" / "t.parquet")
    return root


def _run(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def _write_snapshot(path: Path, digests: dict) -> Path:
    path.write_text(json.dumps({"version": od.SNAPSHOT_VERSION, "roots": [], "exclude": [], "digests": digests}))
    return path


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(PIXELS).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_corrupt(path: Path, variant: int = 0) -> bytes:
    """Write a file that cannot be parsed as its extension says; another ``variant`` is another corrupt file."""
    data = {".parquet": b"PAR1 not a parquet file, take %d" % variant,
            ".json": b'{"not": json, take %d' % variant,
            ".png": _png_bytes()[: 400 + 100 * variant]}[path.suffix]      # truncated inside the image data
    path.write_bytes(data)
    return data


# --------------------------------------------------------------------------
# Walking a tree, keys, diffs
# --------------------------------------------------------------------------

def test_digesting_the_same_tree_twice_gives_identical_digests_and_an_empty_diff(tmp_path):
    tree = _tree(tmp_path / "tree")
    first, second = od.digest_tree(tree), od.digest_tree(tree)
    assert first == second
    assert _rel(first).keys() == TREE_FILES
    assert od.diff_digests(first, second) == {"added": [], "removed": [], "changed": []}


def test_a_copied_tree_has_the_same_digests_under_its_own_root_prefix(tmp_path):
    tree = _tree(tmp_path / "tree")
    copy = tmp_path / "copy"
    shutil.copytree(tree, copy, copy_function=shutil.copyfile)          # fresh mtimes and inodes
    assert _rel(od.digest_tree(copy)) == _rel(od.digest_tree(tree))
    assert all(k.startswith(f"{os.path.abspath(copy)}::") for k in od.digest_tree(copy))


def test_digest_keys_are_the_relative_path_behind_the_absolute_root(tmp_path, monkeypatch):
    _tree(tmp_path / "tree")
    monkeypatch.chdir(tmp_path)
    digests = od.digest_tree("tree")                                    # a relative root: the key is still absolute
    assert set(digests) == {_key("tree", rel) for rel in TREE_FILES}
    assert os.path.isabs(next(iter(digests)).split("::", 1)[0])
    assert list(digests) == sorted(digests)


def test_one_changed_csv_cell_is_reported_as_changed(tmp_path):
    tree = _tree(tmp_path / "tree")
    before = od.digest_tree(tree)
    (tree / "wide.csv").write_text("fly,t0,t1\nA,0.0,1.5\nB,2.5,3.1\n")
    assert od.diff_digests(before, od.digest_tree(tree)) == {
        "added": [], "removed": [], "changed": [_key(tree, "wide.csv")]}


def test_a_csv_is_hashed_as_raw_bytes_so_a_line_ending_change_shows(tmp_path):
    (tmp_path / "a.csv").write_bytes(b"x,y\n1,2\n")
    (tmp_path / "b.csv").write_bytes(b"x,y\r\n1,2\r\n")                  # the same table once parsed
    assert od.digest_file(tmp_path / "a.csv") != od.digest_file(tmp_path / "b.csv")
    assert od.digest_file(tmp_path / "a.csv") == od.digest_file(tmp_path / "a.csv")


def test_a_large_file_is_hashed_across_read_chunks_as_the_sha256_of_its_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(od, "_READ_CHUNK", 16)                          # many chunks, the last one short
    data = bytes(range(256)) * 2 + b"tail"
    path = tmp_path / "big.csv"
    path.write_bytes(data)
    assert od.digest_file(path) == hashlib.sha256(data).hexdigest()
    for position in (0, 16, 100, len(data) - 1):                        # first byte, a chunk start, middle, last
        edited = bytearray(data)
        edited[position] ^= 1
        path.write_bytes(bytes(edited))
        assert od.digest_file(path) != hashlib.sha256(data).hexdigest(), position


def test_file_extensions_are_matched_without_regard_to_case(tmp_path):
    a = _png(tmp_path / "A.PNG", pnginfo=_text_chunks(Software="one"))
    b = _png(tmp_path / "b.Png", pnginfo=_text_chunks(Software="two"), compress_level=0)
    assert a.read_bytes() != b.read_bytes()
    assert od.digest_file(a) == od.digest_file(b)


def test_csv_and_parquet_files_are_never_loaded_with_pandas(tmp_path, monkeypatch):
    import pandas as pd

    def refuse(*args, **kwargs):
        raise AssertionError("pandas must not load an output file")

    for name in ("read_csv", "read_parquet"):
        monkeypatch.setattr(pd, name, refuse)
    digests = od.digest_tree(_tree(tmp_path / "tree"))
    assert len(digests) == len(TREE_FILES)
    assert not [k for k, v in digests.items() if v.startswith("error:")]


def test_added_and_removed_files_are_reported(tmp_path):
    tree = _tree(tmp_path / "tree")
    before = od.digest_tree(tree)
    (tree / "notes.txt").unlink()
    (tree / "sub" / "new.csv").write_text("a\n")
    assert od.diff_digests(before, od.digest_tree(tree)) == {
        "added": [_key(tree, "sub/new.csv")], "removed": [_key(tree, "notes.txt")], "changed": []}


def test_diff_digests_lists_added_removed_and_changed_in_sorted_order():
    a = {"r::b": "1", "r::a": "1", "r::gone": "1", "r::c": "1"}
    b = {"r::c": "2", "r::a": "1", "r::new2": "1", "r::new1": "1", "r::b": "9"}
    assert od.diff_digests(a, b) == {
        "added": ["r::new1", "r::new2"], "removed": ["r::gone"], "changed": ["r::b", "r::c"]}


def test_excluded_globs_are_ignored_at_any_depth(tmp_path):
    tree = _tree(tmp_path / "tree")
    for rel in ("Abdomen-Geometry-all/g.csv", "Abdomen-Geometry/deep/g.csv", "sub/Mounting-Posture/m.png",
                "sub/deeper/Mounting-Posture-x/y/z.csv"):
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_text("x\n")
    unfiltered = od.digest_tree(tree)
    assert len(unfiltered) == len(TREE_FILES) + 4
    filtered = od.digest_tree(tree, exclude=("Abdomen-Geometry*", "Mounting-Posture*"))
    assert _rel(filtered).keys() == TREE_FILES


def test_a_glob_also_matches_the_whole_relative_path_and_a_file_name(tmp_path):
    tree = _tree(tmp_path / "tree")
    (tree / "other").mkdir()
    (tree / "other" / "t.parquet").write_bytes(b"not read: it is excluded below")
    digests = od.digest_tree(tree, exclude=("sub/deeper/*.parquet", "*.txt", "other/"))
    assert _rel(digests).keys() == {"wide.csv", "stats.json", "sub/fig.png"}


def test_digests_do_not_depend_on_the_number_of_workers(tmp_path):
    tree = _tree(tmp_path / "tree")
    for i in range(25):
        (tree / "sub" / f"n{i}.csv").write_text(f"v\n{i}\n")
    assert od.digest_tree(tree, workers=1) == od.digest_tree(tree, workers=4)


def test_at_most_two_parquet_files_are_decoded_at_once(tmp_path, monkeypatch):
    for i in range(8):
        pq.write_table(pa.table({"x": [float(i)] * 10}), tmp_path / f"{i}.parquet")
    lock, active, peak = threading.Lock(), [0], [0]
    pair = threading.Barrier(2, timeout=5)           # a holder waits, slot in hand, until the other holder is in too

    class Spy(pq.ParquetFile):
        def iter_batches(self, *args, **kwargs):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            try:
                pair.wait()                                             # serial digesting breaks this barrier
                yield from super().iter_batches(*args, **kwargs)
            finally:
                with lock:
                    active[0] -= 1

    monkeypatch.setattr(pq, "ParquetFile", Spy)
    digests = od.digest_tree(tmp_path, workers=8)
    assert len(digests) == 8 and not [v for v in digests.values() if v.startswith("error:")]
    assert peak[0] == 2                                                 # parallel, but never more than two


def test_a_missing_or_non_directory_root_is_an_error_not_an_empty_snapshot(tmp_path):
    with pytest.raises(FileNotFoundError):
        od.digest_tree(tmp_path / "nope")
    (tmp_path / "file.csv").write_text("a\n")
    with pytest.raises(NotADirectoryError):
        od.digest_tree(tmp_path / "file.csv")


# --------------------------------------------------------------------------
# Parquet: content, not how it was written
# --------------------------------------------------------------------------

def test_parquet_with_the_same_data_but_different_write_metadata_hashes_equal(tmp_path):
    data = {"x": [1.0, 2.0, float("nan"), 4.0], "s": ["a", "b", None, "d"], "n": [1, 2, 3, 4]}
    plain = pa.table(data)
    one = plain.replace_schema_metadata({"writer": "run-A", "created": "2026-10-13T01:00"})
    two = pa.Table.from_pydict(data, schema=pa.schema([f.with_metadata({"note": "B"}) for f in plain.schema],
                                                      metadata={"writer": "run-B"}))
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    pq.write_table(one, a, compression="snappy", row_group_size=4)
    pq.write_table(two, b, compression="zstd", row_group_size=2)
    assert a.read_bytes() != b.read_bytes()
    assert pq.ParquetFile(a).schema_arrow.metadata != pq.ParquetFile(b).schema_arrow.metadata
    assert od.digest_file(a) == od.digest_file(b)
    assert not od.digest_file(a).startswith("error:")


def test_parquet_digest_does_not_depend_on_the_row_group_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(od, "_PARQUET_BATCH_BYTES", 1)                  # force many small batches from a small table
    monkeypatch.setattr(od, "_MIN_BATCH_ROWS", 7)
    table = pa.table({"x": np.arange(100, dtype="float64"), "s": [str(i) for i in range(100)]})
    digests = set()
    for i, row_group_size in enumerate((100, 33, 10)):
        path = tmp_path / f"{i}.parquet"
        pq.write_table(table, path, row_group_size=row_group_size)
        digests.add(od.digest_file(path))
    assert len(digests) == 1 and not next(iter(digests)).startswith("error:")


def test_parquet_change_in_a_later_batch_is_detected(tmp_path, monkeypatch):
    monkeypatch.setattr(od, "_PARQUET_BATCH_BYTES", 1)                  # 7 rows per batch: 15 batches of 100 rows
    monkeypatch.setattr(od, "_MIN_BATCH_ROWS", 7)
    values = np.arange(100, dtype="float64")
    base = tmp_path / "base.parquet"
    pq.write_table(pa.table({"x": values}), base)
    for row in (6, 7, 55, 98, 99):                                      # end of a batch, start of the next, ...
        edited = values.copy()
        edited[row] += 0.5
        path = tmp_path / f"row{row}.parquet"
        pq.write_table(pa.table({"x": edited}), path)
        assert od.digest_file(path) != od.digest_file(base), row


def test_parquet_with_one_changed_value_hashes_differently(tmp_path):
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    pq.write_table(pa.table({"x": [1.0, 2.0, 3.0], "s": ["a", "b", "c"]}), a)
    pq.write_table(pa.table({"x": [1.0, 2.0, 3.5], "s": ["a", "b", "c"]}), b)
    assert od.digest_file(a) != od.digest_file(b)


def test_parquet_column_order_names_and_types_are_part_of_the_digest(tmp_path):
    base = pa.table({"x": [1, 2], "y": [3.0, 4.0]})
    variants = {
        "base": base,
        "swapped": base.select(["y", "x"]),
        "renamed": base.rename_columns(["x", "z"]),
        "narrower": pa.table({"x": pa.array([1, 2], pa.int32()), "y": [3.0, 4.0]}),
        "fewer_rows": base.slice(0, 1),
        "no_rows": base.slice(0, 0),
    }
    digests = {}
    for name, table in variants.items():
        pq.write_table(table, tmp_path / f"{name}.parquet")
        digests[name] = od.digest_file(tmp_path / f"{name}.parquet")
    assert len(set(digests.values())) == len(variants)


def test_a_parquet_dictionary_column_is_hashed_by_its_values_not_only_its_codes(tmp_path):
    codes = pa.array([0, 1, 0, 1], pa.int32())

    def column(values):
        return pa.DictionaryArray.from_arrays(codes, pa.array(values))

    a, b, plain = (tmp_path / f"{n}.parquet" for n in ("a", "b", "plain"))
    pq.write_table(pa.table({"d": column(["u", "v"])}), a)
    pq.write_table(pa.table({"d": column(["u", "w"])}), b)              # same codes, other dictionary values
    pq.write_table(pa.table({"d": pa.array(["u", "v", "u", "v"])}), plain)
    assert pa.types.is_dictionary(pq.ParquetFile(a).schema_arrow.field("d").type)   # the hazard is really exercised
    assert od.digest_file(a) != od.digest_file(b)
    assert od.digest_file(a) == od.digest_file(plain)                   # the same values, however they are encoded


# --------------------------------------------------------------------------
# PNG: decoded pixels, not file bytes
# --------------------------------------------------------------------------

def test_png_with_the_same_pixels_but_different_text_chunks_and_compression_hashes_equal(tmp_path):
    a = _png(tmp_path / "a.png", pnginfo=_text_chunks(Software="matplotlib 3.9.4", Creation="2026-10-13 01:00"))
    b = _png(tmp_path / "b.png", pnginfo=_text_chunks(Software="matplotlib 3.10", Note="rerun"), compress_level=0)
    assert a.read_bytes() != b.read_bytes()
    assert od.digest_file(a) == od.digest_file(b)
    assert not od.digest_file(a).startswith("error:")


def test_png_with_one_changed_pixel_hashes_differently(tmp_path):
    changed = PIXELS.copy()
    changed[7, 11, 2] ^= 1
    assert od.digest_file(_png(tmp_path / "a.png")) != od.digest_file(_png(tmp_path / "b.png", changed))


def test_png_shape_is_part_of_the_digest(tmp_path):
    flat = np.arange(32, dtype=np.uint8)                                # the same 32 bytes as 2x4 and as 4x2 pixels
    a = _png(tmp_path / "a.png", flat.reshape(2, 4, 4))
    b = _png(tmp_path / "b.png", flat.reshape(4, 2, 4))
    assert od.digest_file(a) != od.digest_file(b)


def test_a_palette_png_is_hashed_by_the_colours_it_draws_not_by_its_palette_indices(tmp_path):
    def palette_png(name, colours, indices):
        image = Image.new("P", (2, 2))
        image.putpalette([c for rgb in colours for c in rgb])
        image.putdata(indices)
        path = tmp_path / name
        image.save(path)
        return path

    red, green, blue = (255, 0, 0), (0, 255, 0), (0, 0, 255)
    a = palette_png("a.png", [red, green], [0, 1, 1, 0])
    b = palette_png("b.png", [green, red], [1, 0, 0, 1])                # other indices, the same picture
    c = palette_png("c.png", [red, blue], [0, 1, 1, 0])                 # one colour changed
    assert od.digest_file(a) == od.digest_file(b)
    assert od.digest_file(a) != od.digest_file(c)


# --------------------------------------------------------------------------
# JSON: canonical dump
# --------------------------------------------------------------------------

def test_json_key_order_and_whitespace_do_not_change_the_digest(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text('{"b": 1, "a": [1, 2, {"y": 2, "x": 1}], "s": "caf\\u00e9"}', encoding="utf-8")
    b.write_text('{\n  "s": "café",\n  "a": [1, 2, {"x": 1, "y": 2}],\n  "b": 1\n}\n', encoding="utf-8")
    assert a.read_bytes() != b.read_bytes()
    assert od.digest_file(a) == od.digest_file(b)
    assert not od.digest_file(a).startswith("error:")


def test_json_with_a_changed_value_or_list_order_hashes_differently(tmp_path):
    texts = {"base": '{"a": [1, 2], "b": 1}', "value": '{"a": [1, 2], "b": 2}', "order": '{"a": [2, 1], "b": 1}',
             "type": '{"a": [1, 2], "b": 1.0}', "key": '{"a": [1, 2], "c": 1}'}
    digests = {}
    for name, text in texts.items():
        (tmp_path / f"{name}.json").write_text(text)
        digests[name] = od.digest_file(tmp_path / f"{name}.json")
    assert len(set(digests.values())) == len(texts)


def test_a_json_file_over_the_parse_limit_is_hashed_as_raw_bytes_instead_of_parsed(tmp_path, monkeypatch):
    monkeypatch.setattr(od, "_JSON_PARSE_LIMIT", 20)
    ordered, reordered, broken = (tmp_path / n for n in ("ordered.json", "reordered.json", "broken.json"))
    ordered.write_text('{"a": 2, "b": 1, "c": [1, 2, 3]}')
    reordered.write_text('{"b": 1, "a": 2, "c": [1, 2, 3]}')
    broken.write_text("{" * 50)
    assert od.digest_file(ordered) == hashlib.sha256(ordered.read_bytes()).hexdigest()
    assert od.digest_file(ordered) != od.digest_file(reordered)         # raw bytes: the key order shows
    assert not od.digest_file(broken).startswith("error:")             # and nothing was parsed


# --------------------------------------------------------------------------
# Bad files never crash a snapshot; a broken environment is not a bad file
# --------------------------------------------------------------------------

def test_corrupt_files_get_an_error_digest_and_the_rest_of_the_tree_is_still_digested(tmp_path):
    tree = _tree(tmp_path / "tree")
    good_png = (tree / "sub" / "fig.png").read_bytes()
    (tree / "bad.parquet").write_bytes(b"PAR1 definitely not a parquet file")
    (tree / "bad.png").write_bytes(good_png[: len(good_png) // 2])
    (tree / "bad.json").write_text("{not json")
    digests = _rel(od.digest_tree(tree))
    for name in ("bad.parquet", "bad.png", "bad.json"):
        assert digests[name].startswith("error:"), name
    assert {k for k, v in digests.items() if not v.startswith("error:")} == TREE_FILES


@pytest.mark.parametrize("name", ["bad.parquet", "bad.png", "bad.json"])
def test_a_corrupt_file_gets_an_error_digest_that_carries_the_sha256_of_its_raw_bytes(tmp_path, name):
    data = _write_corrupt(tmp_path / name)
    marker, error_type, raw_hash = od.digest_file(tmp_path / name).split(":")   # error:<ExceptionType>:<sha256>
    assert marker == "error" and error_type.isidentifier()
    assert raw_hash == hashlib.sha256(data).hexdigest()


def test_two_different_corrupt_parquet_files_compare_as_changed(tmp_path):
    tree = _tree(tmp_path / "tree")
    a, b = tmp_path / "A.json", tmp_path / "B.json"
    _write_corrupt(tree / "bad.parquet", variant=0)
    before = od.digest_tree(tree)
    assert _run("snapshot", "--root", tree, "--out", a).returncode == 0
    _write_corrupt(tree / "bad.parquet", variant=1)                     # still unparsable, but not the same bytes
    after = od.digest_tree(tree)
    assert _run("snapshot", "--root", tree, "--out", b).returncode == 0
    assert before[_key(tree, "bad.parquet")].startswith("error:")
    assert after[_key(tree, "bad.parquet")].startswith("error:")
    assert od.diff_digests(before, after) == {"added": [], "removed": [], "changed": [_key(tree, "bad.parquet")]}
    result = _run("compare", a, b)
    assert result.returncode == 1
    assert "changed: 1" in result.stdout and _key(tree, "bad.parquet") in result.stdout


def test_the_same_corrupt_bytes_in_a_and_b_compare_equal_but_are_still_counted_and_warned(tmp_path):
    tree = _tree(tmp_path / "tree")
    _write_corrupt(tree / "bad.parquet")
    a, b = tmp_path / "A.json", tmp_path / "B.json"
    for snapshot in (a, b):
        taken = _run("snapshot", "--root", tree, "--out", snapshot)
        assert taken.returncode == 0, taken.stderr
        assert "1 unreadable" in taken.stdout                           # counted when it is taken ...
    assert json.loads(a.read_text())["digests"] == json.loads(b.read_text())["digests"]
    result = _run("compare", a, b)
    assert result.returncode == 0                                       # ... the same bytes: nothing changed, but ...
    assert _key(tree, "bad.parquet") in result.stdout                   # ... it is named, ...
    assert "raw bytes identical" in result.stdout                       # ... with what was compared, ...
    assert "1 unreadable" in result.stdout.strip().splitlines()[-1]     # ... and the verdict line says so too


def test_a_corrupt_file_whose_raw_bytes_cannot_be_read_either_keeps_the_bare_error_digest(tmp_path, monkeypatch):
    path = tmp_path / "bad.parquet"
    _write_corrupt(path)

    def unreadable(_path):
        raise OSError("input/output error")

    monkeypatch.setattr(od, "_hash_bytes", unreadable)
    digest = od.digest_file(path)
    assert digest.startswith("error:") and digest.count(":") == 1       # error:<ExceptionType> and no hash


def test_cli_snapshot_counts_unreadable_files_by_exception_type_whatever_their_raw_hash(tmp_path):
    tree = _tree(tmp_path / "tree")
    _write_corrupt(tree / "one.parquet", variant=0)
    _write_corrupt(tree / "two.parquet", variant=1)                     # a different corrupt file: another digest
    out = tmp_path / "A.json"
    result = _run("snapshot", "--root", tree, "--out", out)
    digests = json.loads(out.read_text())["digests"]
    assert digests[_key(tree, "one.parquet")] != digests[_key(tree, "two.parquet")]
    error_type = digests[_key(tree, "one.parquet")].split(":")[1]
    assert f"2 unreadable (error:{error_type} x2)" in result.stdout


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read any file")
@pytest.mark.parametrize("name", ["wide.csv", "sub/deeper/t.parquet"])
def test_a_file_that_cannot_be_opened_gets_the_bare_error_digest(tmp_path, name):
    tree = _tree(tmp_path / "tree")
    (tree / name).chmod(0)
    try:
        digests = _rel(od.digest_tree(tree))
    finally:
        (tree / name).chmod(0o644)
    assert digests[name] == "error:PermissionError"                     # no raw bytes to hash: no hash after the type
    assert {k for k, v in digests.items() if not v.startswith("error:")} == TREE_FILES - {name}


def test_a_file_that_is_not_a_regular_file_gets_an_error_digest_instead_of_blocking(tmp_path):
    fifo = tmp_path / "pipe.csv"
    os.mkfifo(fifo)                                                     # opening a FIFO for reading would block forever
    assert od.digest_file(fifo) == "error:NotARegularFile"


@pytest.mark.parametrize("library, name", [("PIL", "fig.png"), ("pyarrow", "t.parquet")])
def test_a_missing_imaging_or_parquet_library_raises_instead_of_giving_every_file_the_same_error_digest(
        tmp_path, monkeypatch, library, name):
    tree = _tree(tmp_path / "tree")
    monkeypatch.setitem(sys.modules, library, None)                     # makes ``import <library>`` fail
    with pytest.raises(ImportError):
        od.digest_file(next(tree.rglob(name)))
    with pytest.raises(ImportError):
        od.digest_tree(tree)


def test_a_symlinked_file_is_hashed_by_its_content_and_a_dangling_link_gets_an_error_digest(tmp_path):
    tree = _tree(tmp_path / "tree")
    (tree / "link.csv").symlink_to("wide.csv")
    (tree / "dangling.csv").symlink_to("nowhere.csv")
    digests = _rel(od.digest_tree(tree))
    assert digests["link.csv"] == digests["wide.csv"]
    assert digests["dangling.csv"] == "error:FileNotFoundError"


def test_a_symlinked_directory_is_recorded_but_not_followed(tmp_path):
    tree = _tree(tmp_path / "tree")
    (tree / "alias").symlink_to("sub", target_is_directory=True)
    digests = _rel(od.digest_tree(tree))
    assert digests["alias"] == "symlink:sub"
    assert not [k for k in digests if k.startswith("alias/")]
    assert digests.keys() == TREE_FILES | {"alias"}


@pytest.mark.skipif(os.geteuid() == 0, reason="root can list any directory")
def test_a_directory_that_cannot_be_listed_is_recorded_as_an_error_not_silently_skipped(tmp_path):
    tree = _tree(tmp_path / "tree")
    (tree / "locked").mkdir()
    (tree / "locked" / "hidden.csv").write_text("a\n")
    (tree / "locked").chmod(0)
    try:
        digests = _rel(od.digest_tree(tree))
    finally:
        (tree / "locked").chmod(0o755)
    assert digests["locked/"] == "error:PermissionError"
    assert digests.keys() == TREE_FILES | {"locked/"}


# --------------------------------------------------------------------------
# compare: subtract the run-to-run noise
# --------------------------------------------------------------------------

def test_compare_drops_every_path_that_differs_between_a_and_the_noise_snapshot():
    a = {"r::stable": "1", "r::jitter": "1", "r::real": "1", "r::wobble": "1", "r::gone": "1"}
    rerun = {"r::stable": "1", "r::jitter": "2", "r::real": "1", "r::wobble": "2", "r::gone": "1", "r::extra": "7"}
    b = {"r::stable": "1", "r::jitter": "3", "r::real": "2", "r::wobble": "1", "r::extra": "7", "r::brandnew": "5"}
    result = od.compare(a, b, rerun)
    assert result["changed"] == ["r::real"]                              # r::jitter differs run to run: not a finding
    assert result["added"] == ["r::brandnew"]                            # r::extra is in the noise set: not a finding
    assert result["removed"] == ["r::gone"]                              # absent from B, present in both A and A'
    assert result["dropped"] == ["r::extra", "r::jitter"]                # noise that also differs in B
    assert result["noise"] == ["r::extra", "r::jitter", "r::wobble"]     # every path A and A' disagree on


def test_compare_without_a_noise_snapshot_is_the_plain_diff():
    a, b = {"r::a": "1", "r::b": "1"}, {"r::a": "2", "r::c": "1"}
    result = od.compare(a, b)
    assert {k: result[k] for k in ("added", "removed", "changed")} == od.diff_digests(a, b)
    assert result["noise"] == [] and result["dropped"] == []


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def test_cli_snapshot_writes_the_documented_json_and_prints_the_file_count(tmp_path):
    tree = _tree(tmp_path / "tree")
    out = tmp_path / "snaps" / "A.json"                                 # the directory does not exist yet
    result = _run("snapshot", "--root", tree, "--exclude", "*.txt", "--exclude", "deeper", "--out", out)
    assert result.returncode == 0, result.stderr
    assert "3 files" in result.stdout
    snapshot = json.loads(out.read_text())
    assert snapshot["version"] == 2                                     # 2: error digests carry the raw-byte hash
    assert snapshot["roots"] == [os.path.abspath(tree)]
    assert snapshot["exclude"] == ["*.txt", "deeper"]
    assert snapshot["digests"] == od.digest_tree(tree, exclude=("*.txt", "deeper"))
    assert _rel(snapshot["digests"]).keys() == {"wide.csv", "stats.json", "sub/fig.png"}


def test_cli_snapshot_takes_several_roots_into_one_file(tmp_path):
    one, two = _tree(tmp_path / "one"), _tree(tmp_path / "two")
    out = tmp_path / "A.json"
    result = _run("snapshot", "--root", one, "--root", two, "--out", out)
    assert result.returncode == 0, result.stderr
    snapshot = json.loads(out.read_text())
    assert snapshot["roots"] == [os.path.abspath(one), os.path.abspath(two)]
    assert len(snapshot["digests"]) == 2 * len(TREE_FILES)
    assert {k.split("::", 1)[0] for k in snapshot["digests"]} == set(snapshot["roots"])


def test_cli_snapshot_reports_unreadable_files(tmp_path):
    tree = _tree(tmp_path / "tree")
    (tree / "bad.parquet").write_bytes(b"not parquet")
    result = _run("snapshot", "--root", tree, "--out", tmp_path / "A.json")
    assert result.returncode == 0, result.stderr
    assert "1 unreadable" in result.stdout


def test_cli_snapshot_refuses_to_write_inside_a_snapshotted_root(tmp_path):
    tree = _tree(tmp_path / "tree")
    result = _run("snapshot", "--root", tree, "--out", tree / "sub" / "A.json")
    assert result.returncode == 2
    assert not (tree / "sub" / "A.json").exists()


def test_cli_snapshot_refuses_an_out_path_that_is_a_directory_before_digesting(tmp_path):
    tree = _tree(tmp_path / "tree")
    (tmp_path / "snaps").mkdir()
    result = _run("snapshot", "--root", tree, "--out", tmp_path / "snaps")
    assert result.returncode == 2
    assert "[digest]" not in result.stderr                              # failed before any root was walked


def test_cli_compare_exits_zero_for_identical_snapshots_and_one_when_a_file_differs(tmp_path):
    tree = _tree(tmp_path / "tree")
    a, rerun, b = (tmp_path / f"{n}.json" for n in ("A", "A2", "B"))
    assert _run("snapshot", "--root", tree, "--out", a).returncode == 0
    assert _run("snapshot", "--root", tree, "--out", rerun).returncode == 0
    same = _run("compare", a, rerun)
    assert same.returncode == 0, same.stdout + same.stderr

    (tree / "wide.csv").write_text("fly,t0,t1\nA,0.0,1.5\nB,2.5,3.1\n")
    assert _run("snapshot", "--root", tree, "--out", b).returncode == 0
    different = _run("compare", a, b)
    assert different.returncode == 1
    assert _key(tree, "wide.csv") in different.stdout
    assert "changed: 1" in different.stdout


def test_cli_compare_ignores_paths_that_differ_between_a_and_the_noise_snapshot(tmp_path):
    tree = _tree(tmp_path / "tree")
    a, rerun, b = (tmp_path / f"{n}.json" for n in ("A", "A2", "B"))
    assert _run("snapshot", "--root", tree, "--out", a).returncode == 0
    (tree / "notes.txt").write_text("run-to-run jitter, first rerun\n")
    assert _run("snapshot", "--root", tree, "--out", rerun).returncode == 0
    (tree / "notes.txt").write_text("run-to-run jitter, the new config\n")
    assert _run("snapshot", "--root", tree, "--out", b).returncode == 0

    assert _run("compare", a, b).returncode == 1                        # without the noise snapshot it is a difference
    assert _run("compare", a, b, "--noise", rerun).returncode == 0

    (tree / "wide.csv").write_text("fly,t0,t1\nA,9.9,1.5\nB,2.5,3.0\n")  # a real change, on top of the jitter
    assert _run("snapshot", "--root", tree, "--out", b).returncode == 0
    real = _run("compare", a, b, "--noise", rerun)
    assert real.returncode == 1
    assert "changed: 1" in real.stdout                                  # the jitter is not counted ...
    assert _key(tree, "wide.csv") in real.stdout and _key(tree, "notes.txt") not in real.stdout


def test_cli_compare_lists_at_most_fifty_paths_per_category(tmp_path):
    before = {f"/r::f{i:03d}": "a" for i in range(60)}
    after = {key: "b" for key in before}
    result = _run("compare", _write_snapshot(tmp_path / "A.json", before), _write_snapshot(tmp_path / "B.json", after))
    assert result.returncode == 1
    assert "changed: 60" in result.stdout
    assert "/r::f049" in result.stdout and "/r::f050" not in result.stdout
    assert "10 more" in result.stdout


def test_cli_compare_warns_about_paths_that_are_unreadable_in_both_snapshots(tmp_path):
    digests = {"/r::ok.csv": "x", "/r::bad.parquet": "error:ArrowInvalid"}
    a = _write_snapshot(tmp_path / "A.json", digests)
    b = _write_snapshot(tmp_path / "B.json", dict(digests))
    result = _run("compare", a, b)
    assert result.returncode == 0                                       # equal digests: nothing to report ...
    assert "/r::bad.parquet" in result.stdout                           # ... but say what was not compared
    assert "NOT compared" in result.stdout                              # no raw hash after the type: not even the bytes
    verdict = result.stdout.strip().splitlines()[-1]
    assert "1 unreadable" in verdict and "1 of them NOT compared" in verdict    # ... and the verdict line says so too


def test_cli_compare_rejects_a_file_that_is_not_a_current_snapshot(tmp_path):
    good = _write_snapshot(tmp_path / "A.json", {"/r::a": "1"})
    too_old = tmp_path / "old.json"                                     # version 1: its error digests had no raw hash
    too_old.write_text(json.dumps({"version": 1, "digests": {"/r::a": "1"}}))
    too_new = tmp_path / "new.json"
    too_new.write_text(json.dumps({"version": od.SNAPSHOT_VERSION + 1, "digests": {"/r::a": "1"}}))
    not_json = tmp_path / "C.json"
    not_json.write_text("garbage")
    for bad in (too_old, too_new, not_json, tmp_path / "missing.json"):
        result = _run("compare", good, bad)
        assert result.returncode == 2, bad
        assert "Traceback" not in result.stderr
    assert "version 1" in _run("compare", good, too_old).stderr         # and says which version it found
