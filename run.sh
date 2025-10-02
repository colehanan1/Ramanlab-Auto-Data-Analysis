
#!/usr/bin/env bash
set -euo pipefail

# Activate venv
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip -q install --upgrade pip
pip -q install -r requirements.txt

# Run full pipeline
python -m fbpipe.pipeline --config config.yaml all
