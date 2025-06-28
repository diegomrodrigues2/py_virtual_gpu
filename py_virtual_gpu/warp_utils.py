from __future__ import annotations

from threading import Lock
from typing import Any, Dict

from .thread import get_current_thread

__all__ = ["shfl_sync", "ballot_sync"]

# Shared dictionaries keyed by Barrier instances used by the threads. Each entry
# is cleared once all threads of the warp have read the result.
_shfl_values: Dict[int, Dict[int, Any]] = {}
_ballot_values: Dict[int, Dict[int, bool]] = {}
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

    key = id(barrier)
    lane = thread.thread_idx[0]
    with _lock:
        buf = _shfl_values.setdefault(key, {})
        buf[lane] = value
    barrier.wait()
    with _lock:
        result = _shfl_values[key].get(src_lane)
    barrier.wait()
    if lane == 0:
        with _lock:
            _shfl_values.pop(key, None)
    return result


def ballot_sync(predicate: bool) -> int:
    """Collect predicates from all warp threads into a bit mask."""
    thread = get_current_thread()
    if thread is None:
        raise RuntimeError("ballot_sync() must be called from a GPU thread")
    barrier = getattr(thread, "barrier", None)
    if barrier is None:
        raise RuntimeError("ballot_sync() requires a barrier reference")

    key = id(barrier)
    lane = thread.thread_idx[0]
    with _lock:
        buf = _ballot_values.setdefault(key, {})
        buf[lane] = bool(predicate)
    barrier.wait()
    with _lock:
        mask = 0
        for l, pred in _ballot_values[key].items():
            if pred:
                mask |= 1 << l
    barrier.wait()
    if lane == 0:
        with _lock:
            _ballot_values.pop(key, None)
    return mask
