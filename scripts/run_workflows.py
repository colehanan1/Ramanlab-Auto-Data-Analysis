#!/usr/bin/env python3
"""Run the full pipeline plus optional analysis workflows defined in config.yaml."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

# Add project root to sys.path for imports to work
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from scripts.envelope_visuals import (
    EnvelopePlotConfig,
    MatrixPlotConfig,
    generate_envelope_plots,
    generate_reaction_matrices,
)
from scripts.envelope_training import latency_reports
from scripts.envelope_combined import (
    CombineConfig,
    build_wide_csv,
    combine_distance_angle,
    overlay_sources,
    wide_to_matrix,
)


def _resolve_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def _ensure_path(value: str | Path, field: str) -> Path:
    path = _resolve_path(value)
    if path is None:
        raise ValueError(f"Missing required path for '{field}'.")
    return path


def _matrix_plot_config(data: Mapping[str, Any]) -> MatrixPlotConfig:
    opts = dict(data)
    for key in ("matrix_npy", "codes_json", "out_dir"):
        opts[key] = _ensure_path(opts.get(key), key)
    if "trial_orders" in opts and opts["trial_orders"] is not None:
        opts["trial_orders"] = tuple(opts["trial_orders"])
    return MatrixPlotConfig(**opts)  # type: ignore[arg-type]


def _envelope_plot_config(data: Mapping[str, Any]) -> EnvelopePlotConfig:
    opts = dict(data)
    for key in ("matrix_npy", "codes_json", "out_dir"):
        opts[key] = _ensure_path(opts.get(key), key)
    return EnvelopePlotConfig(**opts)  # type: ignore[arg-type]


def _run_envelope_visuals(cfg: Mapping[str, Any] | None) -> None:
    if not cfg:
        return

    matrices_cfg = cfg.get("matrices")
    if matrices_cfg:
        config = _matrix_plot_config(matrices_cfg)
        print(f"[analysis] envelope_visuals.matrices → {config.out_dir}")
        generate_reaction_matrices(config)

    envelopes_cfg = cfg.get("envelopes")
    if envelopes_cfg:
        config = _envelope_plot_config(envelopes_cfg)
        print(f"[analysis] envelope_visuals.envelopes → {config.out_dir}")
        generate_envelope_plots(config)


def _run_training(cfg: Mapping[str, Any] | None) -> None:
    if not cfg:
        return

    envelopes_cfg = cfg.get("envelopes")
    if envelopes_cfg:
        opts = dict(envelopes_cfg)
        opts.setdefault("trial_type", "training")
        config = _envelope_plot_config(opts)
        print(f"[analysis] training.envelopes → {config.out_dir}")
        generate_envelope_plots(config)

    latency_cfg = cfg.get("latency")
    if latency_cfg:
        opts = dict(latency_cfg)
        matrix = _ensure_path(opts.pop("matrix_npy"), "matrix_npy")
        codes = _ensure_path(opts.pop("codes_json"), "codes_json")
        out_dir = _ensure_path(opts.pop("out_dir"), "out_dir")
        trials = tuple(int(t) for t in opts.pop("trials"))
        print(f"[analysis] training.latency → {out_dir}")
        latency_reports(
            matrix,
            codes,
            out_dir,
            trials_of_interest=trials,
            **opts,
        )


def _run_combined(cfg: Mapping[str, Any] | None) -> None:
    if not cfg:
        return

    combine_cfg = cfg.get("combine")
    if combine_cfg:
        opts = dict(combine_cfg)
        opts["root"] = _ensure_path(opts.get("root"), "root")
        if "odor_on" in opts and "odor_on_s" not in opts:
            opts["odor_on_s"] = float(opts.pop("odor_on"))
        if "odor_off" in opts and "odor_off_s" not in opts:
            opts["odor_off_s"] = float(opts.pop("odor_off"))
        config = CombineConfig(**opts)  # type: ignore[arg-type]
        print(f"[analysis] combined.combine → {config.root}")
        combine_distance_angle(config)

    wide_cfg = cfg.get("wide")
    if wide_cfg:
        roots = [str(_ensure_path(root, "roots")) for root in wide_cfg.get("roots", [])]
        if not roots:
            raise ValueError("combined.wide.roots must list at least one directory.")
        output_csv = _ensure_path(wide_cfg.get("output_csv"), "output_csv")
        measure_cols = wide_cfg.get("measure_cols") or ["envelope_of_rms"]
        print(f"[analysis] combined.wide → {output_csv}")
        build_wide_csv(roots, str(output_csv), measure_cols=measure_cols)

    matrix_cfg = cfg.get("matrix")
    if matrix_cfg:
        input_csv = _ensure_path(matrix_cfg.get("input_csv"), "input_csv")
        out_dir = _ensure_path(matrix_cfg.get("out_dir"), "out_dir")
        print(f"[analysis] combined.matrix → {out_dir}")
        wide_to_matrix(str(input_csv), str(out_dir))

    matrices_cfg = cfg.get("matrices")
    if matrices_cfg:
        config = _matrix_plot_config(matrices_cfg)
        print(f"[analysis] combined.matrices → {config.out_dir}")
        generate_reaction_matrices(config)

    envelopes_cfg = cfg.get("envelopes")
    if envelopes_cfg:
        config = _envelope_plot_config(envelopes_cfg)
        print(f"[analysis] combined.envelopes → {config.out_dir}")
        generate_envelope_plots(config)

    overlay_cfg = cfg.get("overlay")
    if overlay_cfg:
        combined_matrix = _ensure_path(overlay_cfg.get("combined_matrix"), "combined_matrix")
        combined_codes = _ensure_path(overlay_cfg.get("combined_codes"), "combined_codes")
        distance_matrix = _ensure_path(overlay_cfg.get("distance_matrix"), "distance_matrix")
        distance_codes = _ensure_path(overlay_cfg.get("distance_codes"), "distance_codes")
        out_dir = _ensure_path(overlay_cfg.get("out_dir"), "out_dir")
        latency_sec = float(overlay_cfg.get("latency_sec", 0.0))
        after_show = float(overlay_cfg.get("after_show_sec", 30.0))
        thresh = float(overlay_cfg.get("threshold_std_mult", 4.0))
        overwrite = bool(overlay_cfg.get("overwrite", False))
        print(f"[analysis] combined.overlay → {out_dir}")
        overlay_sources(
            {
                "RMS × Angle": {
                    "MATRIX_NPY": str(combined_matrix),
                    "CODES_JSON": str(combined_codes),
                },
                "RMS": {
                    "MATRIX_NPY": str(distance_matrix),
                    "CODES_JSON": str(distance_codes),
                },
            },
            latency_sec=latency_sec,
            after_show_sec=after_show,
            output_dir=str(out_dir),
            threshold_mult=thresh,
            overwrite=overwrite,
        )


def _run_pipeline(config_path: Path) -> None:
    cmd = [sys.executable, "-m", "fbpipe.pipeline", "--config", str(config_path), "all"]
    print(f"[analysis] pipeline → {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Pipeline failed with exit code {e.returncode}")
        if e.stderr:
            print(e.stderr.decode() if hasattr(e.stderr, "decode") else e.stderr)
        raise


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to pipeline configuration YAML.")
    args = parser.parse_args(argv)

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    _run_pipeline(config_path)

    analysis_cfg = data.get("analysis") or {}
    _run_envelope_visuals(analysis_cfg.get("envelope_visuals"))
    _run_training(analysis_cfg.get("training"))
    _run_combined(analysis_cfg.get("combined"))


if __name__ == "__main__":  # pragma: no cover
    main()
