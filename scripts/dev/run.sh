#!/usr/bin/env bash
set -euo pipefail

pip -q install --upgrade pip
pip -q install -r requirements.txt

python -m fbpipe.pipeline --config config/config.yaml all
