"""
queue_manager.py - Thread-safe link queue that supports adding YouTube URLs
dynamically while the streaming consumer is actively processing.
"""
import queue
import threading
from typing import Optional


class LinkQueueManager:
    def __init__(self):
        self._queue = queue.Queue()
        self._total_added = 0
        self._lock = threading.Lock()

    def add_link(self, url: str) -> None:
        """Add a YouTube URL to the queue (can be called concurrently from any thread)."""
        if not url:
            return
        url = url.strip().strip('"\'')
        # Skip empty lines, comments, and invalid URLs
        if not url or url.startswith("#"):
            return
        if not (url.startswith("http://") or url.startswith("https://")):
            return
        with self._lock:
            self._queue.put(url)
            self._total_added += 1
            print(f"[QueueManager] Link added: {url} (Total enqueued: {self._total_added})")

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
