"""Memory fence helpers mimicking CUDA semantics."""

from __future__ import annotations

from .thread import get_current_thread

__all__ = [
    "threadfence_block",
    "threadfence",
    "threadfence_system",
]


def threadfence_block() -> None:
    """Fence writes within the current block's shared memory.

    In this simulator the fence simply acquires and releases the
    :class:`SharedMemory` lock of the calling thread to mimic the
    ordering guarantees of ``__threadfence_block`` in CUDA.
    """

    thread = get_current_thread()
    if thread is None:
        raise RuntimeError("threadfence_block() must be called from a GPU thread")
    if getattr(thread, "shared_mem", None) is None:
        return
    with thread.shared_mem.lock:
        pass


def threadfence() -> None:
    """Fence writes to shared and global memory for the whole device."""

    thread = get_current_thread()
    if thread is None:
        raise RuntimeError("threadfence() must be called from a GPU thread")

    if getattr(thread, "global_mem", None) is not None:
        with thread.global_mem.lock:
            pass
    if getattr(thread, "shared_mem", None) is not None:
        with thread.shared_mem.lock:
            pass


def threadfence_system() -> None:
    """Fence writes so that host threads observe them.

    Host and device share memory in this simulator, so this currently
    behaves the same as :func:`threadfence`.
    """

    threadfence()
