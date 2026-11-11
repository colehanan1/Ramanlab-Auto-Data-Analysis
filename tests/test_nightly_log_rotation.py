"""nightly_make_run.sh rolls an oversized cron log before starting the run."""
import datetime
import gzip
import os
import shutil
import subprocess
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "scripts" / "dev" / "nightly_make_run.sh"
MARKER = "STUB-MAKE-RAN"
ERR_MARKER = "STUB-MAKE-STDERR"


def _tree(tmp_path, log_bytes):
    root = tmp_path / "repo"
    (root / "scripts" / "dev").mkdir(parents=True)
    (root / "logs").mkdir()
    shutil.copy(SRC, root / "scripts" / "dev" / "nightly_make_run.sh")
    conda = tmp_path / "conda" / "etc" / "profile.d"
    conda.mkdir(parents=True)
    (conda / "conda.sh").write_text("conda() { return 0; }\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    make = bin_dir / "make"
    make.write_text(f"#!/bin/sh\necho {MARKER}\necho {ERR_MARKER} >&2\n")
    make.chmod(0o755)
    log = root / "logs" / "nightly_make_run.log"
    log.write_bytes(log_bytes)
    return root, log, bin_dir


def _run(tmp_path, root, bin_dir, log, redirect=True):
    env = dict(os.environ)
    env.update(
        CONDA_BASE=str(tmp_path / "conda"),
        PATH=f"{bin_dir}:{env['PATH']}",
        NIGHTLY_LOG_MAX_BYTES="100",
        NIGHTLY_LOG_GZIP_POLL_S="0.2",
    )
    script = root / "scripts" / "dev" / "nightly_make_run.sh"
    target = f">> '{log}' 2>&1" if redirect else f"> '{tmp_path / 'other.out'}' 2>&1"
    subprocess.run(f"/bin/bash '{script}' {target}", shell=True, env=env, check=True, timeout=60)


def _archive(root):
    day = datetime.date.today().isoformat()
    return root / "logs" / "archive" / f"nightly_make_run.{day}.log"


def _wait_gz(path, timeout=15):
    gz = Path(str(path) + ".gz")
    end = time.time() + timeout
    while time.time() < end:
        if gz.exists() and not path.exists():
            return gz
        time.sleep(0.1)
    raise AssertionError(f"{path} was not gzipped within {timeout} s")


def test_small_log_left_alone(tmp_path):
    root, log, bin_dir = _tree(tmp_path, b"tiny\n")
    _run(tmp_path, root, bin_dir, log)
    assert not (root / "logs" / "archive").exists()
    assert log.read_text() == f"tiny\n{MARKER}\n{ERR_MARKER}\n"


def test_oversized_log_rolled(tmp_path):
    body = b"old line\n" * 50
    root, log, bin_dir = _tree(tmp_path, body)
    _run(tmp_path, root, bin_dir, log)
    fresh = log.read_text().splitlines()
    assert fresh[0].startswith("[nightly]") and "nightly_make_run." in fresh[0]
    assert fresh[1:] == [MARKER, ERR_MARKER]     # stdout and stderr both follow the exec
    gz = _wait_gz(_archive(root))
    assert gzip.decompress(gz.read_bytes()) == body


def test_a_writer_still_holding_the_old_log_keeps_its_output(tmp_path):
    """Last night's run still going (or cron's own shell) writes into the moved file
    after the rotation: gzip must wait for it, or that output is lost."""
    body = b"old line\n" * 50
    root, log, bin_dir = _tree(tmp_path, body)
    writer = subprocess.Popen(f"exec >> '{log}'; sleep 1.5; echo LATE-WRITE", shell=True)
    _run(tmp_path, root, bin_dir, log)
    assert writer.wait(10) == 0
    gz = _wait_gz(_archive(root))
    assert gzip.decompress(gz.read_bytes()) == body + b"LATE-WRITE\n"


def test_stdout_not_the_log_does_nothing(tmp_path):
    body = b"old line\n" * 50
    root, log, bin_dir = _tree(tmp_path, body)
    _run(tmp_path, root, bin_dir, log, redirect=False)
    assert log.read_bytes() == body
    assert not (root / "logs" / "archive").exists()
    assert MARKER in (tmp_path / "other.out").read_text()


def test_failing_mv_does_not_stop_run(tmp_path):
    body = b"old line\n" * 50
    root, log, bin_dir = _tree(tmp_path, body)
    mv = bin_dir / "mv"
    mv.write_text("#!/bin/sh\nexit 1\n")
    mv.chmod(0o755)
    _run(tmp_path, root, bin_dir, log)
    assert log.read_bytes() == body + f"{MARKER}\n{ERR_MARKER}\n".encode()


def test_existing_archive_not_overwritten(tmp_path):
    body = b"old line\n" * 50
    root, log, bin_dir = _tree(tmp_path, body)
    arch = _archive(root)
    arch.parent.mkdir()
    arch.write_text("precious\n")
    _run(tmp_path, root, bin_dir, log)
    time.sleep(0.5)
    assert arch.read_text() == "precious\n"
    assert log.read_bytes() == body + f"{MARKER}\n{ERR_MARKER}\n".encode()
