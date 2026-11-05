"""
video_streamer.py - In-memory YouTube video stream decoder.
Pipes direct HTTPS media streams from yt-dlp into FFmpeg stdout as raw RGB24 frames.
Enforces:
  1. Zero disk writes.
  2. Fixed 128x128 scale at 15 FPS.
  3. Dynamic stream URL resolution per video (no expired URLs).
  4. Non-gameplay / black frame filtering.
"""
import os
import re
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

    def _extract_stream_url(self, youtube_url: str) -> tuple[str, float, str]:
        """
        Calls yt-dlp to fetch duration, title, and fresh direct stream HTTPS URL pinned <= 720p.
        Throws RuntimeError if link is invalid or unavailable.
        Returns: (stream_url, duration_seconds, title)
        """
        youtube_url = youtube_url.strip().strip('"\'')
        if not youtube_url or youtube_url.startswith("#") or not (youtube_url.startswith("http://") or youtube_url.startswith("https://")):
            raise ValueError(f"Invalid stream URL '{youtube_url}'")
        cmd = [
            "yt-dlp",
            "-g",
            "--print", "%(duration)s###%(title)s",
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

        lines = [line.strip() for line in proc.stdout.strip().split("\n") if line.strip()]
        duration_sec = 0.0
        title = "Minecraft Gameplay"
        stream_url = ""
        if len(lines) >= 2:
            meta = lines[0].split("###")
            try:
                duration_sec = float(meta[0])
            except (ValueError, IndexError):
                duration_sec = 0.0
            if len(meta) > 1 and meta[1].strip():
                title = meta[1].strip()
            stream_url = lines[1]
        elif len(lines) == 1:
            stream_url = lines[0]

        return stream_url, duration_sec, title

    def _is_local_file(self, source: str) -> bool:
        s = source.strip().strip('"\'')
        if s.startswith("http://") or s.startswith("https://"):
            return False
        if os.path.exists(s) and os.path.isfile(s):
            return True
        ext = os.path.splitext(s)[1].lower()
        return ext in [".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"]

    def _get_local_video_info(self, file_path: str) -> tuple[str, float, str]:
        """
        Extracts duration and filename for a local video file.
        Returns: (file_path, duration_seconds, title)
        """
        file_path = os.path.abspath(file_path.strip().strip('"\''))
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Local video file not found: {file_path}")
        title = os.path.basename(file_path)
        duration_sec = 0.0

        try:
            cmd = [self.ffmpeg_bin, "-nostdin", "-i", file_path, "-hide_banner"]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True, timeout=10)
            import re
            m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", res.stderr)
            if m:
                duration_sec = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        except Exception:
            pass

        return file_path, duration_sec, title

    def stream_frames(self, video_source: str) -> Generator[tuple[np.ndarray, dict], None, None]:
        """
        Yields raw decoded gameplay frames in memory as NumPy arrays with video metadata:
        Handles BOTH YouTube stream URLs and Local video files (.mp4, .mkv, .webm).
        Yields: (frame [128, 128, 3] uint8, metadata dict)
        """
        is_local = self._is_local_file(video_source)
        if is_local:
            try:
                stream_url, duration_sec, title = self._get_local_video_info(video_source)
            except Exception as e:
                print(f"[Streamer Notice] Skipping unavailable local video ({video_source}): {e}", flush=True)
                return
        else:
            try:
                stream_url, duration_sec, title = self._extract_stream_url(video_source)
            except Exception as e:
                print(f"[Streamer Notice] Skipping unavailable YouTube video ({video_source}): {e}", flush=True)
                return

        total_frames = int(duration_sec * self.fps) if duration_sec > 0 else 0
        meta = {
            "title": title,
            "duration_sec": duration_sec,
            "total_frames": total_frames,
            "url": video_source,
            "is_local": is_local
        }

        # FFmpeg subprocess piping directly to stdout (pipe:1) with no disk caching
        if is_local:
            ffmpeg_cmd = [
                self.ffmpeg_bin,
                "-nostdin",
                "-loglevel", "error",
                "-i", stream_url,
                "-vf", f"scale={self.width}:{self.height},fps={self.fps}",
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "pipe:1"
            ]
        else:
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

                yield frame, meta
        finally:
            process.stdout.close()
            process.kill()
            process.wait()
