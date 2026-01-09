
#!/usr/bin/env bash
set -euo pipefail
git init
git add .
git commit -m "Initial commit: fly-behavior-pipeline"
echo "Repo initialized. Add a remote with: git remote add origin <URL> && git push -u origin main"
