.PHONY: help setup run all yolo infer distances plots angles videos clean

help:
        @echo "Targets:"
        @echo "  setup     - install requirements into the active environment"
        @echo "  run       - run the full pipeline with config.yaml"
        @echo "             • override with STEP=<name> or STEPS=\"a b\" to limit pipeline execution"
        @echo "  steps     - list available pipeline steps"
        @echo "  all       - same as run"
        @echo "  clean     - remove build artifacts"

setup:
	pip install --upgrade pip && pip install -r requirements.txt

run:
        python scripts/run_workflows.py --config config.yaml $(foreach s,$(if $(STEPS),$(STEPS),$(STEP)),--pipeline-step $(s))

all: run

steps:
        python scripts/run_workflows.py --config config.yaml --list-steps

yolo:
	python -m fbpipe.steps.yolo_infer --config config.yaml

clean:
	rm -rf **/__pycache__ **/*.pyc
