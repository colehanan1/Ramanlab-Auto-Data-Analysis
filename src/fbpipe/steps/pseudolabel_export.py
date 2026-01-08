from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Sequence

import torch
from ultralytics import YOLO

from ..config import PseudolabelSettings, Settings, get_main_directories, load_settings
from ..utils.columns import EYE_CLASS, PROBOSCIS_CLASS
from ..utils.pseudolabel import (
    discover_videos,
    export_coco_json,
    export_pseudolabel_dataset,
    mine_top_confidence_frames,
)

log = logging.getLogger("fbpipe.pseudolabel_export")
logging.getLogger("ultralytics").setLevel(logging.WARNING)


def _is_cuda_failure(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        tok in msg
        for tok in (
            "cuda",
            "cudnn",
            "cublas",
            "expandable_segments",
            "device-side assert",
            "hip",
        )
    )


def _coerce_names(names_obj, *, min_nc: int) -> List[str]:
    if isinstance(names_obj, dict):
        out: List[Optional[str]] = [None] * max(min_nc, len(names_obj))
        for k, v in names_obj.items():
            try:
                idx = int(k)
            except Exception:
                continue
            if idx < 0:
                continue
            if idx >= len(out):
                out.extend([None] * (idx + 1 - len(out)))
            out[idx] = str(v)
        names = [n if n is not None else f"class{i}" for i, n in enumerate(out)]
        if len(names) > EYE_CLASS and names[EYE_CLASS] == f"class{EYE_CLASS}":
            names[EYE_CLASS] = "eye"
        if len(names) > PROBOSCIS_CLASS and names[PROBOSCIS_CLASS] == f"class{PROBOSCIS_CLASS}":
            names[PROBOSCIS_CLASS] = "proboscis"
        if len(names) < min_nc:
            names.extend([f"class{i}" for i in range(len(names), min_nc)])
        return names[: max(min_nc, len(names))]
    if isinstance(names_obj, (list, tuple)):
        names = [str(x) for x in names_obj]
        if len(names) > EYE_CLASS and names[EYE_CLASS] == f"class{EYE_CLASS}":
            names[EYE_CLASS] = "eye"
        if len(names) > PROBOSCIS_CLASS and names[PROBOSCIS_CLASS] == f"class{PROBOSCIS_CLASS}":
            names[PROBOSCIS_CLASS] = "proboscis"
        if len(names) < min_nc:
            names.extend([f"class{i}" for i in range(len(names), min_nc)])
        return names[: max(min_nc, len(names))]
    names = [f"class{i}" for i in range(min_nc)]
    if len(names) > EYE_CLASS:
        names[EYE_CLASS] = "eye"
    if len(names) > PROBOSCIS_CLASS:
        names[PROBOSCIS_CLASS] = "proboscis"
    return names


def _resolve_pseudolabel_settings(cfg: Settings, args: argparse.Namespace) -> PseudolabelSettings:
    ps = cfg.pseudolabel

    if args.target_total is not None:
        ps = replace(ps, target_total=int(args.target_total))
    if args.stride is not None:
        ps = replace(ps, stride=int(args.stride))
    if args.per_video_cap is not None:
        ps = replace(ps, per_video_cap=int(args.per_video_cap))
    if args.random_sample_per_video is not None:
        ps = replace(ps, random_sample_per_video=int(args.random_sample_per_video))
    if args.batch_size is not None:
        ps = replace(ps, batch_size=int(args.batch_size))
    if args.val_frac is not None:
        ps = replace(ps, val_frac=float(args.val_frac))
    if args.seed is not None:
        ps = replace(ps, seed=int(args.seed))
    if args.image_ext is not None:
        ps = replace(ps, image_ext=str(args.image_ext))
    if args.jpeg_quality is not None:
        ps = replace(ps, jpeg_quality=int(args.jpeg_quality))
    if args.label_format is not None:
        ps = replace(ps, label_format=str(args.label_format))
    if args.max_eye_prob_center_dist_px is not None:
        ps = replace(ps, max_eye_prob_center_dist_px=float(args.max_eye_prob_center_dist_px))
    if args.min_box_area_px is not None:
        ps = replace(ps, min_box_area_px=float(args.min_box_area_px))
    if args.max_box_area_frac is not None:
        ps = replace(ps, max_box_area_frac=float(args.max_box_area_frac))
    if args.reject_multi_eye_first_n_frames is not None:
        ps = replace(ps, reject_multi_eye_first_n_frames=int(args.reject_multi_eye_first_n_frames))
    if args.reject_multi_eye_zero_iou_eps is not None:
        ps = replace(ps, reject_multi_eye_zero_iou_eps=float(args.reject_multi_eye_zero_iou_eps))
    if args.diversity_bins is not None:
        x_bins, y_bins, size_bins, per_bin_cap = (int(v) for v in args.diversity_bins)
        ps = replace(
            ps,
            diversity_bins=replace(
                ps.diversity_bins,
                enabled=True,
                x_bins=x_bins,
                y_bins=y_bins,
                size_bins=size_bins,
                per_bin_cap=per_bin_cap,
            ),
        )

    if args.min_conf is not None:
        ps = replace(ps, min_conf_keep=float(args.min_conf), min_conf_export=float(args.min_conf))
    if args.min_conf_keep is not None:
        ps = replace(ps, min_conf_keep=float(args.min_conf_keep))
    if args.min_conf_export is not None:
        ps = replace(ps, min_conf_export=float(args.min_conf_export))

    if args.dataset_out is not None:
        ps = replace(ps, dataset_out=str(args.dataset_out))
    if getattr(args, "export_coco_json", False):
        ps = replace(ps, export_coco_json=True)

    return ps


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mine top-confidence frames from videos and export pseudo-label YOLO dataset."
    )
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--dataset-out", help="Output dataset directory (default: cfg.pseudolabel.dataset_out)")
    p.add_argument("--roots", nargs="*", help="Override input roots (default: cfg.main_directory)")

    p.add_argument("--target-total", type=int, help="Global top-K target (default: cfg.pseudolabel.target_total)")
    p.add_argument("--per-video-cap", type=int, help="Per-video top-K cap (default: cfg.pseudolabel.per_video_cap)")
    p.add_argument("--stride", type=int, help="Frame stride sampling (default: cfg.pseudolabel.stride)")
    p.add_argument(
        "--random-sample-per-video",
        type=int,
        help="Randomly sample N frames per video (0 disables; default: cfg.pseudolabel.random_sample_per_video)",
    )
    p.add_argument("--batch-size", type=int, help="Inference batch size (default: cfg.pseudolabel.batch_size)")

    p.add_argument("--min-conf", type=float, help="Shorthand: sets both min_conf_keep and min_conf_export")
    p.add_argument("--min-conf-keep", type=float, help="Hard cutoff for candidate consideration")
    p.add_argument("--min-conf-export", type=float, help="Stricter cutoff for writing labels/images")

    p.add_argument("--val-frac", type=float, help="Validation fraction (split by fly)")
    p.add_argument("--seed", type=int, help="Random seed (sampling + split)")

    p.add_argument(
        "--diversity-bins",
        nargs=4,
        metavar=("X_BINS", "Y_BINS", "SIZE_BINS", "PER_BIN_CAP"),
        help="Enable diversity bins with per-bin cap",
    )

    p.add_argument("--max-eye-prob-center-dist-px", type=float, help="Geometry sanity check (0 disables)")
    p.add_argument("--min-box-area-px", type=float, help="Minimum bbox area in pixels (0 disables)")
    p.add_argument("--max-box-area-frac", type=float, help="Maximum bbox area as fraction of image (1 disables)")
    p.add_argument(
        "--reject-multi-eye-first-n-frames",
        type=int,
        help="Skip videos if disjoint multi-eye detected in first N frames (0 disables)",
    )
    p.add_argument(
        "--reject-multi-eye-zero-iou-eps",
        type=float,
        help="IoU tolerance for disjoint multi-eye check",
    )

    p.add_argument("--image-ext", choices=["jpg", "png"], help="Output image format")
    p.add_argument("--jpeg-quality", type=int, help="JPEG quality (when image-ext=jpg)")
    p.add_argument("--label-format", choices=["bbox", "obb"], help="Label format (default: bbox)")

    p.add_argument("--dry-run", action="store_true", help="Write manifest/data.yaml only; no images/labels")
    p.add_argument("--overwrite", action="store_true", help="Overwrite dataset-out if it exists")
    p.add_argument("--export-coco-json", action="store_true", help="Also write COCO JSON annotations")
    p.add_argument("--scan-only", action="store_true", help="Only discover videos and exit")
    return p.parse_args(argv)


def run(
    cfg: Settings,
    *,
    ps: PseudolabelSettings,
    roots: List[Path],
    dataset_out: Path,
    dry_run: bool,
    overwrite: bool,
) -> None:
    log.info("Discovering videos under %d roots...", len(roots))
    videos = discover_videos(roots)
    if not videos:
        fallback = []
        yolo_curation_dirs = getattr(getattr(cfg, "yolo_curation", None), "video_source_dirs", ()) or ()
        if yolo_curation_dirs:
            existing = {p.expanduser().resolve() for p in roots}
            for dir_path in yolo_curation_dirs:
                p = Path(dir_path).expanduser().resolve()
                if p in existing:
                    continue
                fallback.append(p)
        if fallback:
            log.warning(
                "Discovered 0 videos under main roots; retrying with yolo_curation.video_source_dirs (%d dirs).",
                len(fallback),
            )
            videos = discover_videos(fallback)
            roots = fallback

    log.info("Discovered %d videos", len(videos))
    if not videos:
        raise SystemExit(
            "No videos found under provided roots. "
            "If your videos live in secure storage, pass --roots /securedstorage/... "
            "or set yolo_curation.video_source_dirs in config.yaml."
        )

    cuda_available = torch.cuda.is_available()
    use_cuda = cuda_available and not cfg.allow_cpu

    torch.backends.cudnn.benchmark = use_cuda
    if cuda_available:
        torch.backends.cuda.matmul.allow_tf32 = cfg.cuda_allow_tf32
        torch.backends.cudnn.allow_tf32 = cfg.cuda_allow_tf32

    if not cuda_available and not cfg.allow_cpu:
        raise RuntimeError("CUDA is not available. Set allow_cpu: true only for smoke tests.")

    engine_path = str(Path(cfg.model_path).with_suffix(".engine"))
    if Path(engine_path).exists():
        log.info("Loading TensorRT engine: %s", engine_path)
        model = YOLO(engine_path)
    else:
        log.info("Loading PyTorch model: %s", cfg.model_path)
        model = YOLO(cfg.model_path)

    device_in_use: Optional[str] = None

    def _set_device(target: str) -> None:
        nonlocal device_in_use
        if device_in_use == target:
            return
        model.to(target)
        device_in_use = target

    target_device = "cuda" if use_cuda else "cpu"
    try:
        _set_device(target_device)
    except RuntimeError as exc:
        if target_device == "cuda" and cfg.allow_cpu and _is_cuda_failure(exc):
            log.warning("CUDA initialisation failed (%s); falling back to CPU.", exc)
            _set_device("cpu")
        else:
            raise

    if device_in_use is None:
        raise RuntimeError("Failed to initialise YOLO device")

    half = device_in_use == "cuda"

    def predict_fn(frames, conf_thres: float):
        nonlocal device_in_use
        try:
            return model.predict(frames, conf=float(conf_thres), verbose=False, device=device_in_use, half=half)
        except RuntimeError as exc:
            if device_in_use == "cuda" and cfg.allow_cpu and _is_cuda_failure(exc):
                log.warning("CUDA inference failed (%s); switching to CPU.", exc)
                _set_device("cpu")
                return model.predict(frames, conf=float(conf_thres), verbose=False, device=device_in_use, half=False)
            raise

    diversity_bins = None
    if ps.diversity_bins.enabled:
        diversity_bins = (
            int(ps.diversity_bins.x_bins),
            int(ps.diversity_bins.y_bins),
            int(ps.diversity_bins.size_bins),
            int(ps.diversity_bins.per_bin_cap),
        )

    candidates, scan_stats = mine_top_confidence_frames(
        videos,
        predict_fn,
        stride=int(ps.stride),
        random_sample_per_video=int(ps.random_sample_per_video),
        batch_size=int(ps.batch_size),
        target_total=int(ps.target_total),
        per_video_cap=int(ps.per_video_cap),
        min_conf_keep=float(ps.min_conf_keep),
        require_both=bool(ps.require_both),
        export_classes=tuple(int(x) for x in ps.export_classes),
        max_eye_prob_center_dist_px=float(ps.max_eye_prob_center_dist_px),
        min_box_area_px=float(ps.min_box_area_px),
        max_box_area_frac=float(ps.max_box_area_frac),
        reject_multi_eye_first_n_frames=int(ps.reject_multi_eye_first_n_frames),
        reject_multi_eye_zero_iou_eps=float(ps.reject_multi_eye_zero_iou_eps),
        diversity_bins=diversity_bins,
        seed=int(ps.seed),
    )
    log.info("Selected %d candidates (target_total=%d)", len(candidates), ps.target_total)
    for k, v in sorted(scan_stats.items()):
        if k.startswith("reject_"):
            log.info("%s=%d", k, v)

    export_class_ids = set(int(x) for x in ps.export_classes)
    max_class_id = max(export_class_ids) if export_class_ids else 0
    names = _coerce_names(getattr(model, "names", None), min_nc=max_class_id + 1)

    export_stats = export_pseudolabel_dataset(
        candidates,
        dataset_out=dataset_out,
        class_names=names,
        val_frac=float(ps.val_frac),
        seed=int(ps.seed),
        min_conf_export=float(ps.min_conf_export),
        image_ext=str(ps.image_ext),
        jpeg_quality=int(ps.jpeg_quality),
        label_format=str(ps.label_format),
        dry_run=bool(dry_run),
        overwrite=bool(overwrite),
    )

    log.info("Wrote dataset to %s", dataset_out)
    log.info(
        "Export summary: exported=%d train=%d val=%d skipped_existing=%d skipped_export_thres=%d",
        export_stats["exported_images"],
        export_stats["train_images"],
        export_stats["val_images"],
        export_stats["skipped_existing"],
        export_stats["skipped_below_min_conf_export"],
    )

    if ps.export_coco_json:
        train_json, val_json = export_coco_json(dataset_out, class_names=names)
        log.info("Wrote COCO annotations: %s, %s", train_json, val_json)


def main(cfg: Settings) -> None:
    if not getattr(cfg.pseudolabel, "enabled", False):
        log.info("Pseudolabel export is disabled in config (pseudolabel.enabled=false).")
        return
    roots = get_main_directories(cfg)
    dataset_out = (
        Path(cfg.pseudolabel.dataset_out)
        if cfg.pseudolabel.dataset_out
        else (Path(cfg.cache_dir).expanduser().resolve() / "pseudolabel_dataset")
    )
    run(cfg, ps=cfg.pseudolabel, roots=roots, dataset_out=dataset_out, dry_run=False, overwrite=False)


def cli(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_settings(args.config)
    ps = _resolve_pseudolabel_settings(cfg, args)

    roots: List[Path]
    if args.roots:
        roots = [Path(r) for r in args.roots]
    else:
        roots = get_main_directories(cfg)

    dataset_out = (
        Path(ps.dataset_out)
        if ps.dataset_out
        else (Path(cfg.cache_dir).expanduser().resolve() / "pseudolabel_dataset")
    )
    if args.scan_only:
        log.info("Discovering videos under %d roots...", len(roots))
        videos = discover_videos(roots)
        if not videos:
            yolo_curation_dirs = getattr(getattr(cfg, "yolo_curation", None), "video_source_dirs", ()) or ()
            fallback = [Path(p) for p in yolo_curation_dirs] if yolo_curation_dirs else []
            if fallback:
                log.warning(
                    "Discovered 0 videos under main roots; retrying with yolo_curation.video_source_dirs (%d dirs).",
                    len(fallback),
                )
                videos = discover_videos(fallback)
        log.info("Discovered %d videos", len(videos))
        for video in videos[:10]:
            log.info("Video: %s", video)
        return

    run(
        cfg,
        ps=ps,
        roots=roots,
        dataset_out=dataset_out,
        dry_run=bool(args.dry_run),
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    cli()
