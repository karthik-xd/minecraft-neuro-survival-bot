"""
queue_manager.py - Thread-safe link queue that supports adding YouTube URLs
dynamically while the streaming consumer is actively processing.
"""
import os
import queue
import threading
from typing import Optional


class LinkQueueManager:
    def __init__(self):
        self._queue = queue.Queue()
        self._total_added = 0
        self._lock = threading.Lock()

    def add_link(self, url: str) -> None:
        """Add a YouTube URL or local video file path to the queue."""
        if not url:
            return
        url = url.strip().strip('"\'')
        # Skip empty lines, comments, and invalid strings
        if not url or url.startswith("#"):
            return
        is_url = url.startswith("http://") or url.startswith("https://")
        is_local = os.path.exists(url) or any(url.lower().endswith(ext) for ext in [".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"])
        if not (is_url or is_local):
            return
        with self._lock:
            self._queue.put(url)
            self._total_added += 1
            print(f"[QueueManager] Video source added: {url} (Total enqueued: {self._total_added})", flush=True)

    def get_next_link(self, timeout: Optional[float] = None) -> Optional[str]:
        """Fetch the next link. Returns None if empty and timed out."""
        try:
            return self._queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None

    def task_done(self) -> None:
        """Marks the current URL as fully processed."""
        self._queue.task_done()

    def remaining(self) -> int:
        """Returns the number of queued URLs pending processing."""
        return self._queue.qsize()

    def total_added(self) -> int:
        """Returns the total number of links added to this queue manager."""
        return self._total_added
