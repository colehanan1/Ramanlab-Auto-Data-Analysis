.PHONY: help setup run all yolo infer distances plots angles videos clean cron-install cron-uninstall cron-list

help:
	@echo "Targets:"
	@echo "  setup     - install requirements into the active environment"
	@echo "  run       - run the full pipeline with config/config.yaml"
	@echo "  all       - same as run"
	@echo "  cron-install   - schedule nightly run at midnight"
	@echo "  cron-uninstall - remove the scheduled nightly run"
	@echo "  cron-list      - show current crontab entries"
	@echo "  clean     - remove build artifacts"

setup:
	pip install --upgrade pip && pip install -r requirements.txt

run:
	export MPLBACKEND=Agg && python scripts/pipeline/run_workflows.py --config config/config.yaml

all: run

yolo:
	python -m fbpipe.steps.yolo_infer --config config/config.yaml

clean:
	rm -rf **/__pycache__ **/*.pyc

cron-install:
	./scripts/dev/install_midnight_cron.sh

cron-uninstall:
	./scripts/dev/uninstall_midnight_cron.sh

cron-list:
	crontab -l || true
