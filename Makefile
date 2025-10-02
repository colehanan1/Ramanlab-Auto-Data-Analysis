
.PHONY: help setup run all yolo infer distances plots angles videos clean

help:
	@echo "Targets:"
	@echo "  setup     - install requirements into the active environment"
	@echo "  run       - run the full pipeline with config.yaml"
	@echo "  all       - same as run"
	@echo "  clean     - remove build artifacts"

setup:
	pip install --upgrade pip && pip install -r requirements.txt

run: 
	conda activate yolo-env python && -m fbpipe.pipeline --config config.yaml all

all: run

yolo:
	python -m fbpipe.steps.yolo_infer --config config.yaml

clean:
	rm -rf **/__pycache__ **/*.pyc
