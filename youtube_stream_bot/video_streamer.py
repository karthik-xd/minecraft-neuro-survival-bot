"""
video_streamer.py - In-memory YouTube video stream decoder.
Pipes direct HTTPS media streams from yt-dlp into FFmpeg stdout as raw RGB24 frames.
Enforces:
  1. Zero disk writes.
  2. Fixed 128x128 scale at 15 FPS.
  3. Dynamic stream URL resolution per video (no expired URLs).
  4. Non-gameplay / black frame filtering.
"""
import subprocess
import numpy as np
from typing import Generator
import config


class YouTubeFrameStreamer:
    def __init__(self, target_width: int = config.FRAME_WIDTH,
                 target_height: int = config.FRAME_HEIGHT,
                 fps: int = config.TARGET_FPS):
        self.width = target_width
        self.height = target_height
        self.fps = fps
        # Each frame is 128 * 128 * 3 bytes = 49,152 bytes in memory
        self.frame_bytes = self.width * self.height * config.FRAME_CHANNELS
        self.ffmpeg_bin = config.get_ffmpeg_binary()

    def _extract_stream_url(self, youtube_url: str) -> str:
        """
        Calls yt-dlp -g to fetch a fresh direct stream HTTPS URL pinned <= 720p.
        Throws RuntimeError if link is invalid or unavailable.
        """
        youtube_url = youtube_url.strip().strip('"\'')
        if not youtube_url or youtube_url.startswith("#") or not (youtube_url.startswith("http://") or youtube_url.startswith("https://")):
            raise ValueError(f"Invalid stream URL '{youtube_url}'")
        cmd = [
            "yt-dlp",
            "-g",
            "-f", "bestvideo[height<=720][ext=mp4]/bestvideo[height<=720]/best[height<=720]",
            "--no-playlist",
            "--no-warnings",
            youtube_url
        ]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=30
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            raise RuntimeError(f"yt-dlp error: {proc.stderr.strip()[:200]}")

        # The first output line is the video stream URL
        stream_url = proc.stdout.strip().split("\n")[0]
        return stream_url

    def stream_frames(self, youtube_url: str) -> Generator[np.ndarray, None, None]:
        """
        Yields raw decoded gameplay frames in memory as NumPy arrays:
        Shape: [H, W, C] = [128, 128, 3], uint8
        """
        try:
            stream_url = self._extract_stream_url(youtube_url)
        except Exception as e:
            print(f"[Streamer Notice] Skipping unavailable video ({youtube_url}): {e}", flush=True)
            return

        # FFmpeg subprocess piping directly to stdout (pipe:1) with no disk caching
        ffmpeg_cmd = [
            self.ffmpeg_bin,
            "-nostdin",
            "-loglevel", "error",
            "-reconnect", "1",
            "-reconnect_streamed", "1",
            "-reconnect_delay_max", "5",
            "-i", stream_url,
            "-vf", f"scale={self.width}:{self.height},fps={self.fps}",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "pipe:1"
        ]

        process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10 * self.frame_bytes
        )

        try:
            while True:
                # Read exactly one frame's worth of bytes from memory pipe
                raw_bytes = process.stdout.read(self.frame_bytes)
                if len(raw_bytes) < self.frame_bytes:
                    break  # Stream terminated or finished

                # Convert buffer directly to uint8 array [128, 128, 3]
                frame = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((self.height, self.width, 3))

                # Non-gameplay filter: Skip black screens / transition cuts
                if frame.mean() < 8.0:
                    continue

                yield frame
        finally:
            process.stdout.close()
            process.kill()
            process.wait()
