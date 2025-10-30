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

from fbpipe.config import Settings, load_settings
from fbpipe.steps import predict_reactions, reaction_matrix

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
    mirror_directory,
    overlay_sources,
    secure_copy_and_cleanup,
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


def _run_combined(cfg: Mapping[str, Any] | None, settings: Settings | None) -> None:
    if not cfg:
        return

    combine_cfg = cfg.get("combine")
    if combine_cfg:
        opts = dict(combine_cfg)
        roots_cfg = opts.pop("roots", None)
        root_value = opts.pop("root", None)
        if roots_cfg:
            if not isinstance(roots_cfg, Sequence) or isinstance(roots_cfg, (str, bytes, Path)):
                roots_iter = [roots_cfg]
            else:
                roots_iter = list(roots_cfg)
        elif root_value is not None:
            roots_iter = [root_value]
        else:
            raise ValueError("combined.combine requires 'root' or 'roots'.")

        for entry in roots_iter:
            entry_path = _ensure_path(entry, "combine.root")
            run_opts = dict(opts)
            run_opts["root"] = entry_path
            if "odor_on" in run_opts and "odor_on_s" not in run_opts:
                run_opts["odor_on_s"] = float(run_opts.pop("odor_on"))
            if "odor_off" in run_opts and "odor_off_s" not in run_opts:
                run_opts["odor_off_s"] = float(run_opts.pop("odor_off"))
            if "odor_latency" in run_opts and "odor_latency_s" not in run_opts:
                run_opts["odor_latency_s"] = float(run_opts.pop("odor_latency"))
            config = CombineConfig(**run_opts)  # type: ignore[arg-type]
            print(f"[analysis] combined.combine → {config.root}")
            combine_distance_angle(config)

    wide_cfg = cfg.get("wide")
    if wide_cfg:
        mirror_cfg = wide_cfg.get("mirror")
        if mirror_cfg:
            entries = mirror_cfg if isinstance(mirror_cfg, Sequence) else [mirror_cfg]
            for entry in entries:
                src = _ensure_path(entry.get("source"), "mirror.source")
                dest = _ensure_path(entry.get("destination"), "mirror.destination")
                print(f"[analysis] combined.mirror → {dest}")
                copied, bytes_copied = mirror_directory(str(src), str(dest))
                size_mb = bytes_copied / (1024 * 1024) if bytes_copied else 0.0
                print(
                    f"[analysis] combined.mirror copied {copied} file(s) "
                    f"({size_mb:.1f} MiB)."
                )
        roots = [str(_ensure_path(root, "roots")) for root in wide_cfg.get("roots", [])]
        if not roots:
            raise ValueError("combined.wide.roots must list at least one directory.")
        output_csv = _ensure_path(wide_cfg.get("output_csv"), "output_csv")
        measure_cols = wide_cfg.get("measure_cols") or ["envelope_of_rms"]
        fps_fallback = float(wide_cfg.get("fps_fallback", 40.0))
        exclude_cfg = [
            str(_ensure_path(path, "exclude_roots"))
            for path in wide_cfg.get("exclude_roots", [])
        ]
        trial_type_filter = wide_cfg.get("trial_type_filter")
        extra_exports_cfg = wide_cfg.get("trial_type_exports", [])
        extra_exports: dict[str, str] = {}
        if extra_exports_cfg:
            entries = (
                extra_exports_cfg
                if isinstance(extra_exports_cfg, Sequence)
                and not isinstance(extra_exports_cfg, (str, bytes))
                else [extra_exports_cfg]
            )
            for entry in entries:
                if not isinstance(entry, Mapping):
                    raise ValueError("trial_type_exports entries must be mappings.")
                trial_type = entry.get("trial_type")
                export_csv = entry.get("output_csv")
                if not trial_type or not export_csv:
                    raise ValueError(
                        "trial_type_exports entries require 'trial_type' and 'output_csv'."
                    )
                export_path = _ensure_path(export_csv, "trial_type_exports.output_csv")
                extra_exports[str(trial_type).strip().lower()] = str(export_path)
        print(f"[analysis] combined.wide → {output_csv}")
        limits = None
        if settings is not None:
            limits = (settings.class2_min, settings.class2_max)
        build_wide_csv(
            roots,
            str(output_csv),
            measure_cols=measure_cols,
            fps_fallback=fps_fallback,
            exclude_roots=exclude_cfg,
            distance_limits=limits,
            trial_type_filter=trial_type_filter,
            extra_trial_exports=extra_exports or None,
        )

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
        entries = (
            envelopes_cfg
            if isinstance(envelopes_cfg, Sequence)
            and not isinstance(envelopes_cfg, (str, bytes))
            else [envelopes_cfg]
        )
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError("combined.envelopes entries must be mappings.")
            config = _envelope_plot_config(entry)
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
        odor_on = float(overlay_cfg.get("odor_on_s", 30.0))
        odor_off = float(overlay_cfg.get("odor_off_s", 60.0))
        odor_latency = float(overlay_cfg.get("odor_latency_s", overlay_cfg.get("odor_latency", 0.0)))
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
            odor_on_s=odor_on,
            odor_off_s=odor_off,
            odor_latency_s=odor_latency,
            overwrite=overwrite,
        )

    secure_cfg = cfg.get("secure_cleanup")
    if secure_cfg:
        sources = secure_cfg.get("sources") or []
        if not sources:
            raise ValueError("combined.secure_cleanup.sources must list at least one directory.")
        dest = _ensure_path(secure_cfg.get("destination"), "secure_cleanup.destination")
        resolved_sources = [str(_ensure_path(src, "secure_cleanup.sources")) for src in sources]
        perform_cleanup = bool(secure_cfg.get("perform_cleanup", True))
        print(
            f"[analysis] combined.secure_cleanup → dest={dest} sources={resolved_sources} perform_cleanup={perform_cleanup}"
        )
        secure_copy_and_cleanup(
            resolved_sources,
            str(dest),
            perform_cleanup=perform_cleanup,
        )


def _run_reactions(settings: Settings) -> None:
    reaction_cfg = settings.reaction_prediction

    missing = [
        name
        for name, value in (
            ("reaction_prediction.data_csv", reaction_cfg.data_csv),
            ("reaction_prediction.model_path", reaction_cfg.model_path),
            ("reaction_prediction.output_csv", reaction_cfg.output_csv),
        )
        if not value
    ]

    if missing:
        print(
            "[REACTION] Skipping reaction prediction; missing configuration values: "
            + ", ".join(missing)
        )
        return

    print("[analysis] reactions.predict → flybehavior-response predict")
    predict_reactions.main(settings)

    if not reaction_cfg.matrix.out_dir:
        print("[REACTION] Matrix generation skipped; reaction_prediction.matrix.out_dir is empty.")
        return

    print("[analysis] reactions.matrix → reaction_matrix_from_spreadsheet.py")
    reaction_matrix.main(settings)


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

    settings = load_settings(config_path)

    _run_pipeline(config_path)

    analysis_cfg = data.get("analysis") or {}
    _run_combined(analysis_cfg.get("combined"), settings)
    _run_envelope_visuals(analysis_cfg.get("envelope_visuals"))
    _run_training(analysis_cfg.get("training"))
    _run_reactions(settings)


if __name__ == "__main__":  # pragma: no cover
    main()
