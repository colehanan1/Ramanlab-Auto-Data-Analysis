
.PHONY: help setup run all yolo infer distances plots angles videos clean

help:
	@echo "Targets:"
	@echo "  setup     - create .venv and install requirements"
	@echo "  run       - run the full pipeline with config.yaml"
	@echo "  all       - same as run"
	@echo "  clean     - remove build artifacts"

setup:
	python3 -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

run: 
	. .venv/bin/activate && python -m fbpipe.pipeline --config config.yaml all

all: run

clean:
	rm -rf .venv **/__pycache__ **/*.pyc
