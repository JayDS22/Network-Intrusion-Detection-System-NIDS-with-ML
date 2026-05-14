"""Time-bounded sliding window used by the stream processor."""
from __future__ import annotations

from collections import deque
from typing import Any, Callable, Deque, Iterable, List, Optional, Tuple


class SlidingWindow:
    """Holds (timestamp, item) tuples and evicts entries older than span_seconds."""

    def __init__(self, span_seconds: float, max_items: int = 100_000):
        self.span = float(span_seconds)
        self.max  = max_items
        self._buf: Deque[Tuple[float, Any]] = deque(maxlen=max_items)

    def add(self, ts: float, item: Any) -> None:
        self._buf.append((ts, item))
        self._evict(ts)

    def extend(self, pairs: Iterable[Tuple[float, Any]]) -> None:
        for ts, item in pairs:
            self.add(ts, item)

    def _evict(self, now: float) -> None:
        cutoff = now - self.span
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

    def items(self) -> List[Any]:
        return [item for _, item in self._buf]

    def __len__(self) -> int:
        return len(self._buf)

    def aggregate(self, fn: Callable[[List[Any]], Any]) -> Any:
        return fn(self.items())

    def latest(self) -> Optional[Any]:
        return self._buf[-1][1] if self._buf else None
