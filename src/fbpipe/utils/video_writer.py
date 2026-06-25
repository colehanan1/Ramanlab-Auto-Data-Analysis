"""Single-pass H.264 video writer that pipes raw frames straight into ffmpeg.

The old flow encoded every annotated video twice (OpenCV ``VideoWriter`` then a
separate ffmpeg re-encode). Piping raw BGR frames directly into one ffmpeg
``libx264`` process halves that CPU cost and lets us pin the output to
``yuv420p`` + ``+faststart`` so the result is broadly playable.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("fbpipe.video_writer")


class FFmpegFrameWriter:
    """Write BGR uint8 frames to an H.264 mp4 via a single ffmpeg subprocess."""

    def __init__(
        self,
        path: str | Path,
        fps: float,
        width: int,
        height: int,
        crf: int = 30,
        preset: str = "medium",
        ffmpeg: Optional[str] = None,
    ) -> None:
        self.path = str(path)
        self.width = int(width)
        self.height = int(height)
        self.proc: Optional[subprocess.Popen] = None

        ffmpeg_bin = ffmpeg or shutil.which("ffmpeg")
        if not ffmpeg_bin:
            log.warning("ffmpeg not found on PATH; cannot use FFmpegFrameWriter")
            return

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
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # libx264/yuv420p needs even dims
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
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
