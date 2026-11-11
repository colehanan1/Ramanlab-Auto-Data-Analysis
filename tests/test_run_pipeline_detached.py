"""scripts/dev/run_pipeline_detached.sh starts the pipeline as its own systemd user service.

A stub ``systemd-run`` first on PATH records its argv and then runs the command the way
the service would (environment, working directory, output file), so the conda activation
and the arguments run_workflows.py receives are exercised for real. A fake conda env puts
a fake ``python3`` on PATH that records what it was given. Nothing is started.
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "scripts" / "dev" / "run_pipeline_detached.sh"

STUB = r'''#!/bin/bash
printf '%s\n' "$@" > "$RECORD"
[ "${STUB_RC:-0}" != 0 ] && exit "$STUB_RC"
out=
while [ $# -gt 0 ]; do
  case "$1" in
    --setenv=*) export "${1#--setenv=}" ;;
    -p) shift
        case "$1" in
          WorkingDirectory=*) cd "${1#WorkingDirectory=}" ;;
          StandardOutput=append:*) out="${1#StandardOutput=append:}" ;;
        esac ;;
    --*) ;;
    *) break ;;
  esac
  shift
done
exec "$@" >> "$out" 2>&1
'''


@pytest.fixture
def rig(tmp_path):
    root = tmp_path / "repo"
    (root / "scripts" / "dev").mkdir(parents=True)
    script = root / "scripts" / "dev" / "run_pipeline_detached.sh"
    shutil.copy(SRC, script)
    calls = tmp_path / "calls"
    calls.mkdir()
    env_bin = tmp_path / "envs" / "yolo-env" / "bin"
    env_bin.mkdir(parents=True)
    python = env_bin / "python3"
    python.write_text(f'#!/bin/bash\nprintf "%s\\0" "$@" > "{calls}/argv"\n'
                      f'env > "{calls}/env"\npwd > "{calls}/cwd"\necho driver-ran\n')
    python.chmod(0o755)
    conda = tmp_path / "conda"
    (conda / "etc" / "profile.d").mkdir(parents=True)
    (conda / "etc" / "profile.d" / "conda.sh").write_text(
        'conda() { [ "$1" = activate ] || return 0\n'
        f'  echo "$2" > "{calls}/activated"; export PATH="{env_bin}:$PATH"; }}\n')
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "systemd-run").write_text(STUB)
    (bindir / "systemd-run").chmod(0o755)
    record = tmp_path / "systemd-run-argv.txt"

    def run(*args, rc=0):
        env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", CONDA_BASE=str(conda),
                   STUB_RC=str(rc), RECORD=str(record))
        p = subprocess.run(["bash", str(script), *args], env=env, cwd=tmp_path,
                           capture_output=True, text=True)
        argv = record.read_text().splitlines() if record.exists() else []
        return p, argv, root, calls

    return run


def _driver_argv(calls):
    return (calls / "argv").read_text().split("\0")[:-1]


def test_the_driver_gets_the_config_and_every_argument_exactly(rig):
    p, _, root, calls = rig("--thaw", "My Folder", "--x=a b", "$HOME", "pre$$post")
    assert p.returncode == 0, p.stderr
    assert _driver_argv(calls) == ["scripts/pipeline/run_workflows.py", "--config",
                                   "config/config_new.yaml", "--thaw", "My Folder",
                                   "--x=a b", "$HOME", "pre$$post"]
    assert (calls / "cwd").read_text().strip() == str(root)
    assert (calls / "activated").read_text().strip() == "yolo-env"


def test_the_driver_runs_in_the_nightly_environment(rig):
    _, _, _, calls = rig()
    env = dict(line.split("=", 1) for line in (calls / "env").read_text().splitlines()
               if "=" in line)
    for key, value in (("MPLBACKEND", "Agg"), ("ORT_LOGGING_LEVEL", "3"),
                       ("PYTHONNOUSERSITE", "1"), ("PYTHONUNBUFFERED", "1")):
        assert env.get(key) == value, key


def test_a_user_service_that_reports_how_it_ended(rig):
    p, argv, root, _ = rig("--figures-only")
    assert p.returncode == 0, p.stderr
    assert "--user" in argv
    assert "--collect" not in argv               # a failed run stays listed with its result
    assert "--expand-environment=no" in argv     # a "$" in an argument reaches the driver
    unit = [a for a in argv if a.startswith("--unit=")]
    assert len(unit) == 1 and re.fullmatch(r"--unit=fbpipe-run-\d{8}-\d{6}", unit[0])
    for prop in ("Type=exec", "OOMPolicy=continue", "OOMScoreAdjust=100"):
        assert prop in argv, prop
    log = re.search(r"StandardOutput=append:(\S+)", "\n".join(argv)).group(1)
    stop = [a for a in argv if a.startswith("ExecStopPost=")]
    assert len(stop) == 1 and log in stop[0] and "SERVICE_RESULT" in stop[0]
    assert unit[0].split("=", 1)[1] in p.stdout
    assert "systemctl --user stop" in p.stdout


def test_output_goes_to_a_log_under_logs(rig):
    p, argv, root, _ = rig()
    log = Path(re.search(r"StandardOutput=append:(\S+)", "\n".join(argv)).group(1))
    assert log.parent == root / "logs"
    assert re.fullmatch(r"pipeline_detached_\d{8}-\d{6}\.log", log.name)
    assert f"StandardError=append:{log}" in argv
    assert f"WorkingDirectory={root}" in argv
    assert "driver-ran" in log.read_text()
    assert str(log) in p.stdout


def test_failing_systemd_run_is_nonzero(rig):
    p, _, _, calls = rig(rc=3)
    assert p.returncode != 0
    assert "systemd-run failed" in p.stderr
    assert not (calls / "argv").exists()


def test_make_passes_folder_and_args_through():
    p = subprocess.run(["make", "-n", "run-detached", "FOLDER=Hex-Training-24",
                        "ARGS=--figures-only"], cwd=REPO, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "run_pipeline_detached.sh --folder 'Hex-Training-24' --figures-only" in p.stdout
