"""Single-pass H.264 video writer that pipes raw frames straight into ffmpeg.

The old flow encoded every annotated video twice (OpenCV ``VideoWriter`` then a
separate ffmpeg re-encode). Piping raw BGR frames directly into one ffmpeg
process halves that CPU cost and lets us pin the output to ``yuv420p`` +
``+faststart`` so the result is broadly playable.

Encoding defaults to NVIDIA ``h264_nvenc`` (hardware encoder) when available so
the CPU is freed for decode/tracking — important when many videos run in
parallel. Falls back to ``libx264`` (CPU) when NVENC is unavailable.
"""

from __future__ import annotations

import functools
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("fbpipe.video_writer")


@functools.lru_cache(maxsize=4)
def _nvenc_available(ffmpeg_bin: str) -> bool:
    """Probe (once, cached) whether this ffmpeg can actually encode with h264_nvenc.

    Listing the encoder is not enough — the GPU/driver must accept a session — so
    we run a tiny throwaway encode to /dev/null and check the exit code.
    """
    try:
        r = subprocess.run(
            [
                ffmpeg_bin, "-hide_banner", "-loglevel", "error",
                # 256x256: NVENC rejects very small frames ("Frame Dimension less
                # than the minimum supported value"), so probe at a safe size.
                "-f", "lavfi", "-i", "color=c=black:s=256x256:r=30:d=0.1",
                "-c:v", "h264_nvenc", "-pix_fmt", "yuv420p", "-f", "null", "-",
            ],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=30,
        )
        return r.returncode == 0
    except Exception:  # pragma: no cover - depends on local ffmpeg/driver
        return False


class FFmpegFrameWriter:
    """Write BGR uint8 frames to an H.264 mp4 via a single ffmpeg subprocess.

    ``encoder``: ``"auto"`` (default) uses h264_nvenc when available else libx264;
    pass ``"libx264"`` or ``"h264_nvenc"`` to force one.
    """

    def __init__(
        self,
        path: str | Path,
        fps: float,
        width: int,
        height: int,
        crf: int = 30,
        preset: str = "medium",
        ffmpeg: Optional[str] = None,
        encoder: str = "auto",
    ) -> None:
        self.path = str(path)
        self.width = int(width)
        self.height = int(height)
        self.proc: Optional[subprocess.Popen] = None
        self.encoder_used: Optional[str] = None

        ffmpeg_bin = ffmpeg or shutil.which("ffmpeg")
        if not ffmpeg_bin:
            log.warning("ffmpeg not found on PATH; cannot use FFmpegFrameWriter")
            return

        use_nvenc = encoder in ("auto", "nvenc", "h264_nvenc") and _nvenc_available(ffmpeg_bin)
        if encoder == "h264_nvenc" and not use_nvenc:
            log.warning("h264_nvenc requested but unavailable; falling back to libx264")

        if use_nvenc:
            # NVENC: quality via -cq (≈ -crf); GPU preset p1(fastest)..p7(slowest).
            # p4 is a balanced default; encode is on the GPU's dedicated NVENC block.
            codec_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", str(crf)]
            self.encoder_used = "h264_nvenc"
        else:
            codec_args = ["-c:v", "libx264", "-preset", preset, "-crf", str(crf)]
            self.encoder_used = "libx264"

        cmd = [
            ffmpeg_bin,
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", f"{max(float(fps), 1.0):.6f}",
            "-i", "-",
            "-an",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # yuv420p needs even dims
            *codec_args,
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            self.path,
        ]
        try:
            self.proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
        except Exception as exc:  # pragma: no cover - depends on local ffmpeg
            log.warning("Failed to start ffmpeg pipe: %s", exc)
            self.proc = None

    @property
    def ok(self) -> bool:
        return self.proc is not None and self.proc.stdin is not None

    def write(self, frame: np.ndarray) -> bool:
        if not self.ok:
            return False
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
            return True
        except (BrokenPipeError, OSError) as exc:
            log.warning("ffmpeg pipe write failed: %s", exc)
            return False

    def release(self) -> None:
        if self.proc is None:
            return
        # communicate() flushes+closes stdin and drains stderr itself; closing
        # stdin manually first would make it flush a closed pipe and raise.
        try:
            _, stderr = self.proc.communicate(timeout=300)
            if self.proc.returncode not in (0, None) and stderr:
                log.warning("ffmpeg exited %s: %s", self.proc.returncode, stderr.decode(errors="ignore")[:500])
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        finally:
            self.proc = None
