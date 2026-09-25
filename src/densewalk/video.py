"""Video writing that does not destroy the footage.

cv2.VideoWriter with the "mp4v" fourcc encodes MPEG-4 Part 2 at a low default
bitrate. Splat renders are extremely high-frequency, and that combination
macroblocks them into unwatchable garbage regardless of the source quality.
Everything here pipes raw frames to ffmpeg/libx264 at a fixed CRF instead.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


class H264Writer:
    """Drop-in replacement for cv2.VideoWriter: .write(bgr) then .release()."""

    def __init__(self, path: str | Path, width: int, height: int,
                 fps: int = 30, crf: int = 17, preset: str = "slow"):
        self.path = str(path)
        self.proc = subprocess.Popen([
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
            "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", self.path,
        ], stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def release(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()

    # so it can stand in for cv2.VideoWriter where that API is expected
    def isOpened(self) -> bool:  # noqa: N802
        return self.proc.poll() is None
