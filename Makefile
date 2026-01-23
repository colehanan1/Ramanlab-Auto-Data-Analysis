.PHONY: help setup run all yolo infer distances plots angles videos clean cron-install cron-uninstall cron-list backup backup-csvs backup-compressed clean-backups

help:
	@echo "Targets:"
	@echo "  setup     - install requirements into the active environment"
	@echo "  run       - run the full pipeline with config/config.yaml"
	@echo "  all       - same as run"
	@echo "  backup    - run full incremental backup to Box"
	@echo "  backup-csvs - backup only CSVs to Box"
	@echo "  backup-compressed - create compressed emergency archives"
	@echo "  clean-backups - delete old compressed archives"
	@echo "  cron-install   - schedule nightly run at midnight"
	@echo "  cron-uninstall - remove the scheduled nightly run"
	@echo "  cron-list      - show current crontab entries"
	@echo "  clean     - remove build artifacts"

setup:
	pip install --upgrade pip && pip install -r requirements.txt

run: backup-csvs
	export MPLBACKEND=Agg && export ORT_LOGGING_LEVEL=3 && python scripts/pipeline/run_workflows.py --config config/config.yaml
	$(MAKE) backup-csvs

all: run

yolo:
	python -m fbpipe.steps.yolo_infer --config config/config.yaml

backup:
	python scripts/backup_to_box.py

backup-csvs:
	python scripts/backup_to_box.py --csvs-only

backup-compressed:
	python scripts/compress_and_backup.py

clean-backups:
	python scripts/compress_and_backup.py --cleanup-only

clean:
	rm -rf **/__pycache__ **/*.pyc

cron-install:
	./scripts/dev/install_midnight_cron.sh

cron-uninstall:
	./scripts/dev/uninstall_midnight_cron.sh

cron-list:
	crontab -l || true
