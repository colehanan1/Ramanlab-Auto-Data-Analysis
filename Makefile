.PHONY: help setup run all yolo backup clean

help:
	@echo "Targets:"
	@echo "  setup     - install requirements into the active environment"
	@echo "  run       - run full pipeline (includes automatic backups)"
	@echo "  all       - same as run"
	@echo "  backup    - run backups to SMB/Box/Secured (manual only, no auto schedule)"
	@echo "  yolo      - run YOLO inference only"
	@echo "  clean     - remove build artifacts"
	@echo ""
	@echo "Backup behavior:"
	@echo "  - Backups run automatically during 'make run' (before and after pipeline)"
	@echo "  - Compression is DISABLED by default (direct copy of files)"
	@echo "  - Enable compression in config/config.yaml: backups.compression.enabled = true"
	@echo "  - NO automatic cron schedules - backups only run when you execute commands"
	@echo "  - Manual backup: 'make backup'"
	@echo "  - SMB, Box, and Secured storage are all enabled by default"

setup:
	pip install --upgrade pip && pip install -r requirements.txt

run:
	export MPLBACKEND=Agg && export ORT_LOGGING_LEVEL=3 && python scripts/pipeline/run_workflows.py --config config/config.yaml
	$(MAKE) backup

all: run

yolo:
	python -m fbpipe.steps.yolo_infer --config config/config.yaml

backup:
	python scripts/backup_system.py

clean:
	rm -rf **/__pycache__ **/*.pyc
