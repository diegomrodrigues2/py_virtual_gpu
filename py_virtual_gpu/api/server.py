from __future__ import annotations

import sys
import os
import threading
import time
from typing import Callable, Tuple
import subprocess
from pathlib import Path
import shutil

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

    while not getattr(server, "started", True):
        time.sleep(0.05)

    def stop() -> None:
        server.should_exit = True
        thread.join()

    return thread, stop


def start_background_dashboard(
    host: str = "127.0.0.1",
    port: int = 8000,
    app_dir: str | None = None,
) -> tuple[threading.Thread, subprocess.Popen, Callable[[], None]]:
    """Start the API and React dashboard in the background.

    Parameters
    ----------
    host: str
        Interface to bind the API server to.
    port: int
        Port number for the API server.
    app_dir: str, optional
        Path to the React application directory. If not given, the ``app``
        directory at the repository root is used.

    Returns
    -------
    tuple
        ``(api_thread, ui_proc, stop_fn)`` where ``api_thread`` is the running
        API server thread, ``ui_proc`` the ``Popen`` process for ``npm run dev``
        and ``stop_fn`` stops both components.
    """

    api_thread, stop_api = start_background_api(host=host, port=port)

    if app_dir is None:
        app_dir = Path(__file__).resolve().parents[2] / "app"
    else:
        app_dir = Path(app_dir)

    npm_cmd = shutil.which("npm")
    if npm_cmd is None:
        raise RuntimeError(
            "npm executable not found. Please install Node.js and npm to run the dashboard."
        )

    ui_port = 5173
    if ui_port == port:
        ui_port += 1

    cmd = [npm_cmd, "run", "dev", "--", "--port", str(ui_port)]
    if os.name == "nt" and npm_cmd.lower().endswith(".cmd"):
        cmd = ["cmd.exe", "/c"] + cmd

    env = os.environ.copy()
    env["VITE_API_BASE_URL"] = f"http://{host}:{port}"

    # Using pipes for stdout/stderr can cause the npm process to block if the
    # buffers fill up and nothing is reading from them. Redirect the output to
    # ``DEVNULL`` so the dashboard continues to run without hanging.
    ui_proc = subprocess.Popen(
        cmd,
        cwd=app_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )

    def stop() -> None:
        stop_api()
        ui_proc.terminate()
        try:
            ui_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            ui_proc.kill()

    return api_thread, ui_proc, stop
