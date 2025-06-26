from __future__ import annotations

import sys
import threading
from typing import Callable, Tuple

import uvicorn


def _running_in_notebook() -> bool:
    """Return True if running inside a Jupyter notebook."""
    return "ipykernel" in sys.modules


def start_background_api(host: str = "127.0.0.1", port: int = 8000) -> Tuple[threading.Thread, Callable[[], None]]:
    """Start the FastAPI app in a background thread.

    Parameters
    ----------
    host: str
        Interface to bind the server to.
    port: int
        Port number for the server.

    Returns
    -------
    tuple
        ``(thread, stop_fn)`` where ``thread`` is the running server thread and
        ``stop_fn`` stops the server and joins the thread.
    """

    access_log = not _running_in_notebook()
    config = uvicorn.Config(
        "py_virtual_gpu.api.main:app",
        host=host,
        port=port,
        log_level="info",
        access_log=access_log,
    )
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    def stop() -> None:
        server.should_exit = True
        thread.join()

    return thread, stop
