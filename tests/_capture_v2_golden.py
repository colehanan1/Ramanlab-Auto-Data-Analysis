"""Regenerate the v2 goldens under tests/golden/v2/.

Run ONLY when a v2 behavior change is intentional:
    python tests/_capture_v2_golden.py
The protocol-switch v2 guard (test_protocol_v2_golden.py) compares against these.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from scripts.analysis import envelope_combined as ec  # noqa: E402
from scripts.analysis import envelope_visuals as ev  # noqa: E402
from _protocol_fixtures import build_envelope_fixture, MEASURE_COL  # noqa: E402


def main() -> None:
    golden_dir = ROOT / "tests" / "golden" / "v2"
    golden_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        dataset_root = build_envelope_fixture(td)
        ev.set_protocol("v2")
        wide_csv = td / "wide.csv"
        ec.build_wide_csv([str(dataset_root)], str(wide_csv), measure_cols=[MEASURE_COL])
        matrix_dir = td / "matrix"
        ec.wide_to_matrix(str(wide_csv), str(matrix_dir))
        shutil.copy(wide_csv, golden_dir / "wide.csv")
        shutil.copy(matrix_dir / "envelope_matrix_float16.npy", golden_dir / "envelope_matrix_float16.npy")
        shutil.copy(matrix_dir / "code_maps.json", golden_dir / "code_maps.json")
    print(f"Wrote goldens to {golden_dir}: {sorted(p.name for p in golden_dir.iterdir())}")


if __name__ == "__main__":
    main()
