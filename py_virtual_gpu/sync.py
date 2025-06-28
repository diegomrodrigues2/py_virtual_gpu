from .thread import get_current_thread


def syncthreads() -> None:
    """Synchronize all threads in the current block."""
    thread = get_current_thread()
    if thread is None:
        raise RuntimeError("syncthreads() must be called from a GPU thread")
    thread.barrier.wait()


__all__ = ["syncthreads"]
