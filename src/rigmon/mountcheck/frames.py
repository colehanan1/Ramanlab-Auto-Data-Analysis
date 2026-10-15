"""Read frames from a rig's MJPEG /preview stream (read-only; the Pi is untouched)."""

from __future__ import annotations

import time
from typing import Callable, Iterable, Iterator

import cv2
import httpx
import numpy as np

SOI, EOI = b"\xff\xd8", b"\xff\xd9"


def iter_jpegs(chunks: Iterable[bytes]) -> Iterator[bytes]:
    buf = b""
    for chunk in chunks:
        buf += chunk
        while True:
            s = buf.find(SOI)
            if s < 0:
                buf = buf[-1:]
                break
            e = buf.find(EOI, s + 2)
            if e < 0:
                buf = buf[s:]
                break
            yield buf[s:e + 2]
            buf = buf[e + 2:]


def crop_square(img: np.ndarray) -> np.ndarray:
    """Keep the left ``height`` columns: the Pi preview carries a stride strip on the right."""
    h, w = img.shape[:2]
    return img[:, :h] if w > h else img


def decode(jpg: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("undecodable JPEG")
    return crop_square(img)


def sample_preview(url: str, seconds: float, fps: float, *, stop: Callable[[], bool] = lambda: False,
                   timeout: float = 10.0, clock=time.monotonic) -> tuple[list[bytes], bool]:
    """Keep ~``fps`` frames/s for ``seconds``. Returns (jpegs, completed). Never raises
    for network trouble: an unreachable or silent stream gives ([], False)."""
    frames: list[bytes] = []
    start = clock()
    end, next_t = start + seconds, start
    try:
        with httpx.Client(timeout=httpx.Timeout(timeout)) as client:
            with client.stream("GET", url) as resp:
                resp.raise_for_status()
                for jpg in iter_jpegs(resp.iter_bytes()):
                    now = clock()
                    if now >= end:
                        return frames, True
                    if stop():
                        return frames, False
                    if now >= next_t:
                        frames.append(jpg)
                        next_t += 1.0 / fps
    except httpx.HTTPError:
        pass
    return frames, False
