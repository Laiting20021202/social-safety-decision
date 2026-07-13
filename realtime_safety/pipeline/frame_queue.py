from __future__ import annotations

import queue
import threading
from typing import Generic, TypeVar

T = TypeVar("T")


class LatestQueue(Generic[T]):
    """Bounded queue that drops the oldest value when full."""

    def __init__(self, maxsize: int = 2) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be positive")
        self._queue: queue.Queue[T] = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self._dropped = 0

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    @property
    def maxsize(self) -> int:
        return self._queue.maxsize

    def qsize(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()

    def put_latest(self, item: T) -> None:
        with self._lock:
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                    self._dropped += 1
                except queue.Empty:
                    pass
            self._queue.put_nowait(item)

    def get(self, timeout: float | None = None) -> T:
        return self._queue.get(timeout=timeout)

    def get_latest(self, timeout: float | None = None) -> T:
        item = self._queue.get(timeout=timeout)
        self._queue.task_done()
        while True:
            try:
                newer = self._queue.get_nowait()
                self._queue.task_done()
                item = newer
                with self._lock:
                    self._dropped += 1
            except queue.Empty:
                return item

    def clear(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                return
