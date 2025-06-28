from __future__ import annotations

from multiprocessing import Lock
from typing import Any

from .thread import get_current_thread

__all__ = ["shfl_sync", "ballot_sync"]

_lock = Lock()


def shfl_sync(value: Any, src_lane: int) -> Any:
    """Exchange ``value`` across threads and return ``src_lane``'s value.

    All threads belonging to the same warp call this function providing their
    local ``value``. The function waits for every thread to contribute and then
    returns the value provided by ``src_lane``.
    """
    thread = get_current_thread()
    if thread is None:
        raise RuntimeError("shfl_sync() must be called from a GPU thread")
    barrier = getattr(thread, "barrier", None)
    if barrier is None:
        raise RuntimeError("shfl_sync() requires a barrier reference")

    buf = getattr(thread, "warp_buffer", None)
    if buf is None:
        raise RuntimeError("shfl_sync() requires a warp_buffer reference")
    lane = thread.thread_idx[0]
    with _lock:
        buf[lane] = value
    barrier.wait()
    with _lock:
        result = buf[src_lane]
    barrier.wait()
    return result


def ballot_sync(predicate: bool) -> int:
    """Collect predicates from all warp threads into a bit mask."""
    thread = get_current_thread()
    if thread is None:
        raise RuntimeError("ballot_sync() must be called from a GPU thread")
    barrier = getattr(thread, "barrier", None)
    if barrier is None:
        raise RuntimeError("ballot_sync() requires a barrier reference")

    buf = getattr(thread, "warp_buffer", None)
    if buf is None:
        raise RuntimeError("ballot_sync() requires a warp_buffer reference")
    lane = thread.thread_idx[0]
    with _lock:
        buf[lane] = 1 if predicate else 0
    barrier.wait()
    with _lock:
        if lane == 0:
            mask = 0
            for i in range(len(buf)):
                if buf[i]:
                    mask |= 1 << i
            buf[0] = mask
    barrier.wait()
    mask = buf[0]
    return mask
