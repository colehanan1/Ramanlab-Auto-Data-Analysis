from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .columns import EYE_CLASS, PROBOSCIS_CLASS
from .multi_fly import enforce_zero_iou_and_topk
from .vision import xyxy_to_cxcywh
from .yolo_results import collect_detections

log = logging.getLogger("fbpipe.pseudolabel")

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v"}


def _slugify(text: str, *, max_len: int = 120) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return slug[:max_len] if max_len and len(slug) > max_len else slug


def xyxy_to_yolo_norm(
    xyxy: Sequence[float], width_px: int, height_px: int, *, clip: bool = True
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    if width_px <= 0 or height_px <= 0:
        raise ValueError(f"Invalid image size: {width_px}x{height_px}")
    if clip:
        x1 = max(0.0, min(x1, float(width_px)))
        x2 = max(0.0, min(x2, float(width_px)))
        y1 = max(0.0, min(y1, float(height_px)))
        y2 = max(0.0, min(y2, float(height_px)))
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return (
        float(np.clip(cx / width_px, 0.0, 1.0)),
        float(np.clip(cy / height_px, 0.0, 1.0)),
        float(np.clip(w / width_px, 0.0, 1.0)),
        float(np.clip(h / height_px, 0.0, 1.0)),
    )


def write_yolo_bbox_label_file(
    txt_path: Path,
    labels: Sequence[Tuple[int, Sequence[float]]],
    *,
    width_px: int,
    height_px: int,
) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        for class_id, xyxy in labels:
            cx, cy, w, h = xyxy_to_yolo_norm(xyxy, width_px, height_px)
            f.write(f"{int(class_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def write_yolo_obb_label_file(
    txt_path: Path,
    labels: Sequence[Tuple[int, Sequence[float]]],
    *,
    width_px: int,
    height_px: int,
) -> None:
    """Write Ultralytics OBB labels as 4 corner points (xyxyxyxy) normalized to [0,1]."""

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        for class_id, corners_xyxyxyxy in labels:
            corners = list(float(v) for v in corners_xyxyxyxy)
            if len(corners) != 8:
                raise ValueError(f"Expected 8 floats for xyxyxyxy, got {len(corners)}")
            norm: List[float] = []
            for i, v in enumerate(corners):
                if i % 2 == 0:
                    norm.append(float(np.clip(v / width_px, 0.0, 1.0)))
                else:
                    norm.append(float(np.clip(v / height_px, 0.0, 1.0)))
            f.write(
                f"{int(class_id)} "
                + " ".join(f"{v:.6f}" for v in norm)
                + "\n"
            )


def discover_videos(roots: Sequence[Path]) -> List[Path]:
    """Discover videos under dataset roots.

    The pipeline convention is: ``root/<fly_dir>/*.mp4``. If a root itself
    contains videos, it is treated as a fly directory.
    """

    seen: set[Path] = set()
    videos: List[Path] = []
    for root in roots:
        root = root.expanduser()
        if not root.exists():
            log.warning("Root does not exist: %s", root)
            continue

        root_videos = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
        if root_videos:
            for p in root_videos:
                resolved = p.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                videos.append(resolved)
            continue

        for fly_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            if any(token in fly_dir.name.lower() for token in ("plot", "summary", "threshold")):
                continue
            for p in fly_dir.iterdir():
                if not p.is_file():
                    continue
                if p.suffix.lower() not in VIDEO_EXTS:
                    continue
                resolved = p.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                videos.append(resolved)

    videos.sort(key=lambda p: str(p))
    return videos


@dataclass(frozen=True)
class FrameCandidate:
    source_video: Path
    frame_idx: int
    fly_id: str
    width_px: int
    height_px: int
    eye_conf: float
    prob_conf: float
    eye_bbox_xyxy: Tuple[float, float, float, float]
    prob_bbox_xyxy: Tuple[float, float, float, float]
    selection_score: float
    eye_corners_xyxyxyxy: Optional[Tuple[float, ...]] = None
    prob_corners_xyxyxyxy: Optional[Tuple[float, ...]] = None
    unique_key: str = ""


def _candidate_key(video_path: Path, frame_idx: int) -> str:
    return f"{video_path.resolve()}#frame={int(frame_idx)}"


def _push_topk_heap(
    heap: List[Tuple[float, str, FrameCandidate]],
    candidate: FrameCandidate,
    cap: int,
) -> None:
    if cap <= 0:
        return
    item = (float(candidate.selection_score), candidate.unique_key, candidate)
    if len(heap) < cap:
        import heapq

        heapq.heappush(heap, item)
        return

    if item <= heap[0]:
        return

    import heapq

    heapq.heapreplace(heap, item)


def _best_det_for_class(dets: Dict[int, Dict[str, object]], class_id: int):
    cls_data = dets.get(int(class_id))
    if not cls_data:
        return None
    boxes = cls_data.get("boxes")
    scores = cls_data.get("scores")
    corners = cls_data.get("corners")
    if boxes is None or scores is None:
        return None
    boxes_arr = np.asarray(boxes)
    scores_arr = np.asarray(scores)
    if boxes_arr.shape[0] == 0:
        return None
    best_idx = int(np.argmax(scores_arr))
    box = tuple(float(v) for v in boxes_arr[best_idx].tolist())
    conf = float(scores_arr[best_idx])
    corner_points = None
    if isinstance(corners, list) and best_idx < len(corners):
        c = corners[best_idx]
        if c is not None:
            flat = [float(v) for pt in c for v in pt]
            if len(flat) == 8:
                corner_points = tuple(flat)
    return conf, box, corner_points


def _bbox_area_px(xyxy: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _passes_geometry_sanity(
    eye_xyxy: Tuple[float, float, float, float],
    prob_xyxy: Tuple[float, float, float, float],
    *,
    max_center_dist_px: float,
) -> bool:
    if max_center_dist_px <= 0:
        return True
    eye_cx, eye_cy, _, _ = xyxy_to_cxcywh(np.array(eye_xyxy, dtype=np.float32))
    prob_cx, prob_cy, _, _ = xyxy_to_cxcywh(np.array(prob_xyxy, dtype=np.float32))
    dx = float(prob_cx - eye_cx)
    dy = float(prob_cy - eye_cy)
    return (dx * dx + dy * dy) <= float(max_center_dist_px) ** 2


def _bin_key(
    candidate: FrameCandidate,
    *,
    x_bins: int,
    y_bins: int,
    size_bins: int,
) -> Tuple[int, int, int]:
    eye_cx, eye_cy, _, _ = xyxy_to_cxcywh(np.array(candidate.eye_bbox_xyxy, dtype=np.float32))
    x = float(np.clip(eye_cx / candidate.width_px, 0.0, 0.999999))
    y = float(np.clip(eye_cy / candidate.height_px, 0.0, 0.999999))
    area_frac = float(
        np.clip(
            _bbox_area_px(candidate.prob_bbox_xyxy) / float(candidate.width_px * candidate.height_px),
            0.0,
            0.999999,
        )
    )
    xb = int(x * x_bins) if x_bins > 0 else 0
    yb = int(y * y_bins) if y_bins > 0 else 0
    sb = int(area_frac * size_bins) if size_bins > 0 else 0
    return xb, yb, sb


def mine_top_confidence_frames(
    video_paths: Sequence[Path],
    predict_fn: Callable[[Sequence[np.ndarray], float], Sequence[object]],
    *,
    stride: int,
    random_sample_per_video: int,
    batch_size: int,
    target_total: int,
    per_video_cap: int,
    min_conf_keep: float,
    require_both: bool = True,
    export_classes: Tuple[int, ...] = (EYE_CLASS, PROBOSCIS_CLASS),
    max_eye_prob_center_dist_px: float = 0.0,
    min_box_area_px: float = 0.0,
    max_box_area_frac: float = 1.0,
    diversity_bins: Optional[Tuple[int, int, int, int]] = None,
    reject_multi_eye_first_n_frames: int = 5,
    reject_multi_eye_zero_iou_eps: float = 1e-9,
    seed: int = 1337,
) -> Tuple[List[FrameCandidate], Dict[str, int]]:
    """Scan videos and retain a bounded top-K set of high-confidence frames."""

    if stride <= 0 and random_sample_per_video <= 0:
        raise ValueError("Either stride must be >0 or random_sample_per_video must be >0.")
    if target_total <= 0:
        raise ValueError("target_total must be > 0")
    if per_video_cap <= 0:
        raise ValueError("per_video_cap must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    rng = random.Random(seed)
    stats: Dict[str, int] = {
        "videos_total": len(video_paths),
        "videos_open_failed": 0,
        "videos_skipped_multi_eye": 0,
        "frames_sampled": 0,
        "frames_rejected": 0,
        "frames_kept": 0,
    }
    rejection_counts: Dict[str, int] = {}

    global_heap: List[Tuple[float, str, FrameCandidate]] = []
    bins: Dict[Tuple[int, int, int], List[Tuple[float, str, FrameCandidate]]] = {}
    use_bins = diversity_bins is not None and len(diversity_bins) == 4 and diversity_bins[3] > 0
    collect_classes = tuple(sorted(set(int(c) for c in export_classes) | {EYE_CLASS, PROBOSCIS_CLASS}))

    def _result_has_disjoint_multi_eye(result: object) -> bool:
        dets = collect_detections(result, collect_classes)
        eye_data = dets.get(EYE_CLASS)
        if not eye_data:
            return False
        boxes = np.asarray(eye_data.get("boxes"))
        scores = np.asarray(eye_data.get("scores"))
        if boxes.size == 0 or scores.size == 0:
            return False
        keep = scores >= float(min_conf_keep)
        if int(np.count_nonzero(keep)) < 2:
            return False
        kept_boxes = boxes[keep]
        kept_scores = scores[keep]
        kept_boxes, _ = enforce_zero_iou_and_topk(
            kept_boxes,
            kept_scores,
            k=2,
            eps=float(reject_multi_eye_zero_iou_eps),
        )
        return bool(kept_boxes.shape[0] >= 2)

    def _batch_has_disjoint_multi_eye(frames: Sequence[np.ndarray]) -> bool:
        if not frames:
            return False
        results = list(predict_fn(frames, float(min_conf_keep)))
        if len(results) != len(frames):
            raise RuntimeError(f"predict_fn returned {len(results)} results for {len(frames)} frames")
        return any(_result_has_disjoint_multi_eye(result) for result in results)

    for vid_idx, video_path in enumerate(video_paths, start=1):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            stats["videos_open_failed"] += 1
            log.warning("Cannot open video: %s", video_path)
            continue

        fly_id = video_path.parent.name
        width_px = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height_px = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if width_px <= 0 or height_px <= 0:
            width_px = 1080
            height_px = 1080

        if reject_multi_eye_first_n_frames > 0:
            max_check_frames = int(reject_multi_eye_first_n_frames)
            if total_frames > 0:
                max_check_frames = min(max_check_frames, int(total_frames))
            if max_check_frames > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                check_batch: List[np.ndarray] = []
                checked = 0
                skip_video = False
                while checked < max_check_frames:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    check_batch.append(frame)
                    checked += 1
                    if len(check_batch) >= batch_size:
                        if _batch_has_disjoint_multi_eye(check_batch):
                            skip_video = True
                            break
                        check_batch = []
                if not skip_video and check_batch:
                    if _batch_has_disjoint_multi_eye(check_batch):
                        skip_video = True
                if skip_video:
                    stats["videos_skipped_multi_eye"] += 1
                    log.warning(
                        "Skipping video due to disjoint multi-eye in first %d frames: %s",
                        max_check_frames,
                        video_path,
                    )
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        per_video_heap: List[Tuple[float, str, FrameCandidate]] = []

        def handle_batch(frame_items: List[Tuple[int, np.ndarray]]):
            if not frame_items:
                return
            frames = [img for _, img in frame_items]
            results = list(predict_fn(frames, float(min_conf_keep)))
            if len(results) != len(frame_items):
                raise RuntimeError(
                    f"predict_fn returned {len(results)} results for {len(frame_items)} frames"
                )
            for (frame_idx, _img), result in zip(frame_items, results):
                dets = collect_detections(result, collect_classes)
                eye_det = _best_det_for_class(dets, EYE_CLASS)
                prob_det = _best_det_for_class(dets, PROBOSCIS_CLASS)

                def reject(reason: str):
                    stats["frames_rejected"] += 1
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

                if eye_det is None:
                    reject("missing_eye")
                    continue
                if prob_det is None:
                    reject("missing_proboscis")
                    continue

                eye_conf, eye_xyxy, eye_corners = eye_det
                prob_conf, prob_xyxy, prob_corners = prob_det

                if require_both and (eye_conf < min_conf_keep or prob_conf < min_conf_keep):
                    reject("below_min_conf_keep")
                    continue

                if not _passes_geometry_sanity(
                    eye_xyxy, prob_xyxy, max_center_dist_px=max_eye_prob_center_dist_px
                ):
                    reject("geometry_sanity")
                    continue

                if min_box_area_px > 0.0:
                    if _bbox_area_px(eye_xyxy) < min_box_area_px:
                        reject("min_area_eye")
                        continue
                    if _bbox_area_px(prob_xyxy) < min_box_area_px:
                        reject("min_area_proboscis")
                        continue

                if max_box_area_frac < 1.0:
                    max_area_px = float(max_box_area_frac) * float(width_px * height_px)
                    if _bbox_area_px(eye_xyxy) > max_area_px:
                        reject("max_area_eye")
                        continue
                    if _bbox_area_px(prob_xyxy) > max_area_px:
                        reject("max_area_proboscis")
                        continue

                score = min(float(eye_conf), float(prob_conf))
                candidate = FrameCandidate(
                    source_video=video_path,
                    frame_idx=int(frame_idx),
                    fly_id=fly_id,
                    width_px=width_px,
                    height_px=height_px,
                    eye_conf=float(eye_conf),
                    prob_conf=float(prob_conf),
                    eye_bbox_xyxy=tuple(float(v) for v in eye_xyxy),
                    prob_bbox_xyxy=tuple(float(v) for v in prob_xyxy),
                    selection_score=score,
                    eye_corners_xyxyxyxy=eye_corners,
                    prob_corners_xyxyxyxy=prob_corners,
                    unique_key=_candidate_key(video_path, int(frame_idx)),
                )
                _push_topk_heap(per_video_heap, candidate, per_video_cap)
                stats["frames_kept"] += 1

        frame_batch: List[Tuple[int, np.ndarray]] = []

        if random_sample_per_video > 0 and total_frames > 0:
            k = min(int(random_sample_per_video), int(total_frames))
            indices = rng.sample(range(total_frames), k=k)
            indices.sort()
            for frame_idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ok, frame = cap.read()
                if not ok:
                    rejection_counts["read_failed"] = rejection_counts.get("read_failed", 0) + 1
                    continue
                stats["frames_sampled"] += 1
                frame_batch.append((int(frame_idx), frame))
                if len(frame_batch) >= batch_size:
                    handle_batch(frame_batch)
                    frame_batch = []
        else:
            if stride <= 0:
                cap.release()
                raise ValueError("stride must be >0 when random_sample_per_video is disabled")
            frame_idx = 0
            while True:
                ok = cap.grab()
                if not ok:
                    break
                if frame_idx % stride != 0:
                    frame_idx += 1
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    break
                stats["frames_sampled"] += 1
                frame_batch.append((int(frame_idx), frame))
                if len(frame_batch) >= batch_size:
                    handle_batch(frame_batch)
                    frame_batch = []
                frame_idx += 1

        if frame_batch:
            handle_batch(frame_batch)

        cap.release()

        if use_bins:
            x_bins, y_bins, size_bins, per_bin_cap = diversity_bins  # type: ignore[misc]
            for _, _, cand in per_video_heap:
                bkey = _bin_key(cand, x_bins=x_bins, y_bins=y_bins, size_bins=size_bins)
                heap = bins.setdefault(bkey, [])
                _push_topk_heap(heap, cand, int(per_bin_cap))
        else:
            for _, _, cand in per_video_heap:
                _push_topk_heap(global_heap, cand, target_total)

        if vid_idx % 25 == 0:
            kept = len(global_heap) if not use_bins else sum(len(h) for h in bins.values())
            cutoff = 0.0
            if not use_bins and len(global_heap) >= target_total and global_heap:
                cutoff = float(global_heap[0][0])
            log.info(
                "Scanned %d/%d videos | sampled=%d kept=%d | cutoff=%.3f",
                vid_idx,
                len(video_paths),
                stats["frames_sampled"],
                kept,
                cutoff,
            )

    # Attach rejection detail counts
    stats.update({f"reject_{k}": v for k, v in rejection_counts.items()})

    if use_bins:
        candidates = [cand for heap in bins.values() for _, _, cand in heap]
        global_heap = []
        for cand in candidates:
            _push_topk_heap(global_heap, cand, target_total)

    selected = [cand for _, _, cand in sorted(global_heap, key=lambda t: (-t[0], t[1]))]
    return selected, stats


def split_train_val_by_fly(
    candidates: Sequence[FrameCandidate], *, val_frac: float, seed: int
) -> Tuple[set[str], set[str]]:
    fly_ids = sorted({c.fly_id for c in candidates})
    if not fly_ids:
        return set(), set()
    if val_frac <= 0.0:
        return set(fly_ids), set()
    if val_frac >= 1.0:
        return set(), set(fly_ids)
    rng = random.Random(int(seed))
    rng.shuffle(fly_ids)
    n_val = int(round(len(fly_ids) * float(val_frac)))
    if n_val == 0 and val_frac > 0:
        n_val = 1
    if n_val >= len(fly_ids):
        n_val = len(fly_ids) - 1
    val_ids = set(fly_ids[:n_val])
    train_ids = set(fly_ids[n_val:])
    return train_ids, val_ids


def _load_existing_manifest_keys(manifest_csv: Path) -> set[str]:
    if not manifest_csv.exists():
        return set()
    keys: set[str] = set()
    with open(manifest_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row.get("source_video", "")
            frame = row.get("frame_idx", "")
            if not src or not frame:
                continue
            try:
                frame_idx = int(frame)
            except ValueError:
                continue
            keys.add(_candidate_key(Path(src), frame_idx))
    return keys


def export_pseudolabel_dataset(
    candidates: Sequence[FrameCandidate],
    *,
    dataset_out: Path,
    class_names: Sequence[str],
    val_frac: float,
    seed: int,
    min_conf_export: float,
    image_ext: str = "jpg",
    jpeg_quality: int = 95,
    label_format: str = "bbox",
    dry_run: bool = False,
    overwrite: bool = False,
) -> Dict[str, int]:
    """Materialize an Ultralytics-compatible dataset folder from candidates."""

    dataset_out = dataset_out.expanduser().resolve()
    if dataset_out.exists():
        if overwrite:
            log.warning("Overwriting existing dataset_out: %s", dataset_out)
            shutil.rmtree(dataset_out)
        else:
            manifest_csv = dataset_out / "manifest.csv"
            if not manifest_csv.exists():
                raise FileExistsError(
                    f"dataset_out exists but no manifest.csv found: {dataset_out} (use --overwrite)"
                )

    images_train = dataset_out / "images" / "train"
    images_val = dataset_out / "images" / "val"
    labels_train = dataset_out / "labels" / "train"
    labels_val = dataset_out / "labels" / "val"
    for p in (images_train, images_val, labels_train, labels_val):
        p.mkdir(parents=True, exist_ok=True)

    manifest_csv = dataset_out / "manifest.csv"
    existing_keys = set()
    if manifest_csv.exists() and not overwrite:
        existing_keys = _load_existing_manifest_keys(manifest_csv)
        if existing_keys:
            log.info("Resuming: %d items already in manifest.csv", len(existing_keys))

    train_ids, val_ids = split_train_val_by_fly(candidates, val_frac=val_frac, seed=seed)

    # Write data.yaml
    nc = len(class_names)
    data_yaml = dataset_out / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {dataset_out}",
                "train: images/train",
                "val: images/val",
                f"nc: {nc}",
                "names:",
                *[f"  - {name}" for name in class_names],
                "",
            ]
        ),
        encoding="utf-8",
    )

    stats: Dict[str, int] = {
        "selected_total": len(candidates),
        "exported_images": 0,
        "skipped_existing": 0,
        "skipped_below_min_conf_export": 0,
        "skipped_read_failed": 0,
        "written_manifest_rows": 0,
        "train_images": 0,
        "val_images": 0,
    }

    write_header = not manifest_csv.exists() or overwrite
    with open(manifest_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_video",
                "frame_idx",
                "out_image_path",
                "out_label_path",
                "eye_conf",
                "prob_conf",
                "eye_bbox_xyxy",
                "prob_bbox_xyxy",
                "width_px",
                "height_px",
                "selection_score",
                "rejection_reason",
            ],
        )
        if write_header:
            writer.writeheader()

        # Group for efficient extraction
        by_video: Dict[Path, List[FrameCandidate]] = {}
        for cand in candidates:
            by_video.setdefault(cand.source_video, []).append(cand)

        for video_path, items in sorted(by_video.items(), key=lambda kv: str(kv[0])):
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                log.warning("Cannot reopen video for export: %s", video_path)
                for cand in items:
                    writer.writerow(
                        {
                            "source_video": str(video_path.resolve()),
                            "frame_idx": cand.frame_idx,
                            "out_image_path": "",
                            "out_label_path": "",
                            "eye_conf": cand.eye_conf,
                            "prob_conf": cand.prob_conf,
                            "eye_bbox_xyxy": json.dumps(cand.eye_bbox_xyxy),
                            "prob_bbox_xyxy": json.dumps(cand.prob_bbox_xyxy),
                            "width_px": cand.width_px,
                            "height_px": cand.height_px,
                            "selection_score": cand.selection_score,
                            "rejection_reason": "video_open_failed",
                        }
                    )
                    stats["written_manifest_rows"] += 1
                continue

            for cand in sorted(items, key=lambda c: c.frame_idx):
                key = cand.unique_key or _candidate_key(cand.source_video, cand.frame_idx)
                if key in existing_keys:
                    stats["skipped_existing"] += 1
                    continue

                if cand.selection_score < float(min_conf_export):
                    stats["skipped_below_min_conf_export"] += 1
                    writer.writerow(
                        {
                            "source_video": str(video_path.resolve()),
                            "frame_idx": cand.frame_idx,
                            "out_image_path": "",
                            "out_label_path": "",
                            "eye_conf": cand.eye_conf,
                            "prob_conf": cand.prob_conf,
                            "eye_bbox_xyxy": json.dumps(cand.eye_bbox_xyxy),
                            "prob_bbox_xyxy": json.dumps(cand.prob_bbox_xyxy),
                            "width_px": cand.width_px,
                            "height_px": cand.height_px,
                            "selection_score": cand.selection_score,
                            "rejection_reason": "below_min_conf_export",
                        }
                    )
                    stats["written_manifest_rows"] += 1
                    continue

                split = "val" if cand.fly_id in val_ids else "train"
                stem = _slugify(
                    f"{cand.fly_id}__{video_path.stem}__f{cand.frame_idx:06d}__"
                    f"{hashlib.sha1(str(video_path.resolve()).encode('utf-8')).hexdigest()[:10]}"
                )
                out_img_rel = Path("images") / split / f"{stem}.{image_ext.lstrip('.')}"
                out_lbl_rel = Path("labels") / split / f"{stem}.txt"
                out_img_abs = dataset_out / out_img_rel
                out_lbl_abs = dataset_out / out_lbl_rel

                if not dry_run:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cand.frame_idx))
                    ok, frame = cap.read()
                    if not ok:
                        stats["skipped_read_failed"] += 1
                        writer.writerow(
                            {
                                "source_video": str(video_path.resolve()),
                                "frame_idx": cand.frame_idx,
                                "out_image_path": "",
                                "out_label_path": "",
                                "eye_conf": cand.eye_conf,
                                "prob_conf": cand.prob_conf,
                                "eye_bbox_xyxy": json.dumps(cand.eye_bbox_xyxy),
                                "prob_bbox_xyxy": json.dumps(cand.prob_bbox_xyxy),
                                "width_px": cand.width_px,
                                "height_px": cand.height_px,
                                "selection_score": cand.selection_score,
                                "rejection_reason": "read_failed",
                            }
                        )
                        stats["written_manifest_rows"] += 1
                        continue

                    out_img_abs.parent.mkdir(parents=True, exist_ok=True)
                    params: List[int] = []
                    if image_ext.lower() in {"jpg", "jpeg"}:
                        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
                    cv2.imwrite(str(out_img_abs), frame, params)

                    if label_format.lower() == "obb":
                        labels = []
                        if cand.eye_corners_xyxyxyxy is not None:
                            labels.append((EYE_CLASS, cand.eye_corners_xyxyxyxy))
                        if cand.prob_corners_xyxyxyxy is not None:
                            labels.append((PROBOSCIS_CLASS, cand.prob_corners_xyxyxyxy))
                        if not labels:
                            write_yolo_bbox_label_file(
                                out_lbl_abs,
                                [
                                    (EYE_CLASS, cand.eye_bbox_xyxy),
                                    (PROBOSCIS_CLASS, cand.prob_bbox_xyxy),
                                ],
                                width_px=cand.width_px,
                                height_px=cand.height_px,
                            )
                        else:
                            write_yolo_obb_label_file(
                                out_lbl_abs,
                                labels,
                                width_px=cand.width_px,
                                height_px=cand.height_px,
                            )
                    else:
                        write_yolo_bbox_label_file(
                            out_lbl_abs,
                            [
                                (EYE_CLASS, cand.eye_bbox_xyxy),
                                (PROBOSCIS_CLASS, cand.prob_bbox_xyxy),
                            ],
                            width_px=cand.width_px,
                            height_px=cand.height_px,
                        )

                writer.writerow(
                    {
                        "source_video": str(video_path.resolve()),
                        "frame_idx": cand.frame_idx,
                        "out_image_path": str(out_img_rel),
                        "out_label_path": str(out_lbl_rel),
                        "eye_conf": cand.eye_conf,
                        "prob_conf": cand.prob_conf,
                        "eye_bbox_xyxy": json.dumps(cand.eye_bbox_xyxy),
                        "prob_bbox_xyxy": json.dumps(cand.prob_bbox_xyxy),
                        "width_px": cand.width_px,
                        "height_px": cand.height_px,
                        "selection_score": cand.selection_score,
                        "rejection_reason": "",
                    }
                )
                stats["written_manifest_rows"] += 1
                stats["exported_images"] += 1
                if split == "val":
                    stats["val_images"] += 1
                else:
                    stats["train_images"] += 1

            cap.release()

    return stats


def export_coco_json(
    dataset_out: Path,
    *,
    class_names: Sequence[str],
    manifest_csv: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Export minimal COCO JSON annotations from ``manifest.csv``.

    Writes:
      - dataset_out/annotations_train.json
      - dataset_out/annotations_val.json
    """

    dataset_out = dataset_out.expanduser().resolve()
    manifest_csv = manifest_csv or (dataset_out / "manifest.csv")
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Missing manifest.csv at {manifest_csv}")

    categories = [{"id": int(i), "name": str(name)} for i, name in enumerate(class_names)]

    def coco_dict():
        return {"images": [], "annotations": [], "categories": categories}

    coco_train = coco_dict()
    coco_val = coco_dict()
    image_id = 1
    ann_id = 1

    with open(manifest_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("rejection_reason"):
                continue
            out_image_path = row.get("out_image_path", "")
            if not out_image_path:
                continue
            if out_image_path.startswith("images/train/"):
                coco = coco_train
            elif out_image_path.startswith("images/val/"):
                coco = coco_val
            else:
                continue

            try:
                width_px = int(float(row.get("width_px", "0")))
                height_px = int(float(row.get("height_px", "0")))
            except ValueError:
                continue

            coco["images"].append(
                {
                    "id": image_id,
                    "file_name": out_image_path,
                    "width": width_px,
                    "height": height_px,
                }
            )

            def add_ann(category_id: int, xyxy_json: str):
                nonlocal ann_id
                try:
                    x1, y1, x2, y2 = (float(v) for v in json.loads(xyxy_json))
                except Exception:
                    return
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": int(category_id),
                        "bbox": [x1, y1, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

            add_ann(EYE_CLASS, row.get("eye_bbox_xyxy", "[]"))
            add_ann(PROBOSCIS_CLASS, row.get("prob_bbox_xyxy", "[]"))

            image_id += 1

    train_path = dataset_out / "annotations_train.json"
    val_path = dataset_out / "annotations_val.json"
    train_path.write_text(json.dumps(coco_train, indent=2), encoding="utf-8")
    val_path.write_text(json.dumps(coco_val, indent=2), encoding="utf-8")
    return train_path, val_path


__all__ = [
    "FrameCandidate",
    "discover_videos",
    "export_coco_json",
    "export_pseudolabel_dataset",
    "mine_top_confidence_frames",
    "split_train_val_by_fly",
    "write_yolo_bbox_label_file",
    "write_yolo_obb_label_file",
    "xyxy_to_yolo_norm",
]
