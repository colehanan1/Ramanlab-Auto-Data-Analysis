"""Every rig script takes --n-flies (1-4, default 4) and publishes/records it."""
from pathlib import Path

import pytest

PICODE = Path(__file__).resolve().parent.parent / "PiCode"
SCRIPTS = ["combinedv2_1.py", "combinedv2_1_pi1.py", "combinedv2_1_pi3.py"]


@pytest.mark.parametrize("script", SCRIPTS)
def test_n_flies_flag(script):
    src = (PICODE / script).read_text()
    assert '"--n-flies"' in src and "choices=(1, 2, 3, 4)" in src and "default=4" in src
    assert "n_flies=args.n_flies" in src          # PHASE.set_session -> /status.json
    assert "Flies mounted:" in src                 # session header line


def test_status_page_shows_fly_count():
    page = (PICODE / "phase_status.py").read_text()
    assert "ses.n_flies" in page and '" flies"' in page
