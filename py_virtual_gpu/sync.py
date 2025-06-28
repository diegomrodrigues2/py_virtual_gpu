from threading import BrokenBarrierError
from .thread import get_current_thread
from .errors import SynchronizationError


def syncthreads() -> None:
    """Synchronize all threads in the current block."""
    thread = get_current_thread()
    if thread is None:
        raise RuntimeError("syncthreads() must be called from a GPU thread")
    block = getattr(thread, "block", None)
    if block is not None:
        block.barrier_sync()
        return
    timeout = getattr(thread, "barrier_timeout", None)
    try:
        thread.barrier.wait(timeout=timeout)
    except BrokenBarrierError as exc:
        raise SynchronizationError("Barrier wait timed out") from exc

def sync_grid() -> None:
    """Synchronize all threads in the current grid."""

    thread = get_current_thread()
    if thread is None:
        raise RuntimeError("sync_grid() must be called from a GPU thread")
    barrier = getattr(thread, "grid_barrier", None)
    if barrier is None:
        raise RuntimeError("Kernel was not launched cooperatively")
    timeout = getattr(thread, "grid_barrier_timeout", None)
    try:
        barrier.wait(timeout=timeout)
    except BrokenBarrierError as exc:
        raise SynchronizationError("Barrier wait timed out") from exc


__all__ = ["syncthreads", "sync_grid"]
