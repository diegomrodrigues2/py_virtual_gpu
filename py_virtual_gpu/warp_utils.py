from __future__ import annotations

from multiprocessing import Lock
from multiprocessing.managers import SyncManager
from typing import Any
import os
import json

from .thread import get_current_thread

__all__ = ["shfl_sync", "ballot_sync"]

# Shared dictionaries keyed by Barrier instances used by the threads. Each entry
# is cleared once all threads of the warp have read the result.
_manager = None  # type: ignore[var-annotated]
_shfl_values = None  # type: ignore[var-annotated]
_ballot_values = None  # type: ignore[var-annotated]
_lock = Lock()


def _ensure_manager() -> None:
    """Lazily create or connect to the shared ``SyncManager`` instance."""
    global _manager, _shfl_values, _ballot_values
    if _manager is not None:
        return

    addr_json = os.environ.get("PYVGPU_MANAGER_ADDRESS")
    auth_hex = os.environ.get("PYVGPU_MANAGER_AUTH")
    if addr_json is None or auth_hex is None:
        # Main process: start a new manager and expose its address
        manager = SyncManager()
        manager.start()
        address = manager.address
        os.environ["PYVGPU_MANAGER_ADDRESS"] = json.dumps(address)
        os.environ["PYVGPU_MANAGER_AUTH"] = manager._authkey.hex()
        _manager = manager
    else:
        address = json.loads(addr_json)
        authkey = bytes.fromhex(auth_hex)
        manager = SyncManager(address=tuple(address) if isinstance(address, list) else address, authkey=authkey)
        manager.connect()
        _manager = manager

    _shfl_values = _manager.dict()  # type: ignore[assignment]
    _ballot_values = _manager.dict()  # type: ignore[assignment]


def shfl_sync(value: Any, src_lane: int) -> Any:
    """Exchange ``value`` across threads and return ``src_lane``'s value.

    All threads belonging to the same warp call this function providing their
    local ``value``. The function waits for every thread to contribute and then
    returns the value provided by ``src_lane``.
    """
    _ensure_manager()
    thread = get_current_thread()
    if thread is None:
        raise RuntimeError("shfl_sync() must be called from a GPU thread")
    barrier = getattr(thread, "barrier", None)
    if barrier is None:
        raise RuntimeError("shfl_sync() requires a barrier reference")

    key = id(barrier)
    lane = thread.thread_idx[0]
    with _lock:
        if key not in _shfl_values:  # type: ignore[operator]
            _shfl_values[key] = _manager.dict()  # type: ignore[index]
        buf = _shfl_values[key]
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
    _ensure_manager()
    thread = get_current_thread()
    if thread is None:
        raise RuntimeError("ballot_sync() must be called from a GPU thread")
    barrier = getattr(thread, "barrier", None)
    if barrier is None:
        raise RuntimeError("ballot_sync() requires a barrier reference")

    key = id(barrier)
    lane = thread.thread_idx[0]
    with _lock:
        if key not in _ballot_values:  # type: ignore[operator]
            _ballot_values[key] = _manager.dict()  # type: ignore[index]
        buf = _ballot_values[key]
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
