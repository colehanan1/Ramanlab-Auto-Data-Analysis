
#!/usr/bin/env bash
set -euo pipefail

pip -q install --upgrade pip
pip -q install -r requirements.txt

# Run full pipeline
python -m fbpipe.pipeline --config config.yaml all
