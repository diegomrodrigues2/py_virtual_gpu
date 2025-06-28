from threading import BrokenBarrierError
from .thread import get_current_thread
from .errors import SynchronizationError


def syncthreads() -> None:
    """Synchronize all threads in the current block."""
    thread = get_current_thread()
    if thread is None:
        raise RuntimeError("syncthreads() must be called from a GPU thread")
    timeout = getattr(thread, "barrier_timeout", None)
    try:
        thread.barrier.wait(timeout=timeout)
    except BrokenBarrierError as exc:
        raise SynchronizationError("Barrier wait timed out") from exc


__all__ = ["syncthreads"]
