import os
import sys
import time
import httpx
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu.api.server import start_background_api, start_background_dashboard


def test_start_background_api():
    thread, stop = start_background_api(port=8001)
    try:
        for _ in range(50):
            try:
                resp = httpx.get("http://127.0.0.1:8001/status", timeout=1.0)
                if resp.status_code == 200:
                    break
            except Exception:
                time.sleep(0.1)
        else:
            pytest.fail("API server did not start")
        assert thread.is_alive()
    finally:
        stop()
        assert not thread.is_alive()


def test_start_background_api_suppresses_access_logs(monkeypatch):
    """When running in a notebook, access logs should be disabled."""

    captured_access_log = {}

    class DummyConfig:
        def __init__(self, *args, **kwargs):
            captured_access_log['value'] = kwargs.get('access_log')
            self.access_log = kwargs.get('access_log')

    class DummyServer:
        def __init__(self, config):
            self.config = config
            self.should_exit = False

        def run(self):
            return

    monkeypatch.setitem(sys.modules, 'ipykernel', object())
    monkeypatch.setattr(start_background_api.__globals__['uvicorn'], 'Config', DummyConfig)
    monkeypatch.setattr(start_background_api.__globals__['uvicorn'], 'Server', DummyServer)

    thread, stop = start_background_api(port=8002)
    try:
        stop()
        thread.join(timeout=1)
    finally:
        monkeypatch.setitem(sys.modules, 'ipykernel', None)

    assert captured_access_log.get('value') is False


def test_start_background_dashboard(monkeypatch):
    class DummyProcess:
        def __init__(self):
            self.terminated = False

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return

    dummy_proc = DummyProcess()
    monkeypatch.setattr(
        start_background_dashboard.__globals__["subprocess"],
        "Popen",
        lambda *a, **kw: dummy_proc,
    )

    thread, proc, stop = start_background_dashboard(port=8003)
    try:
        for _ in range(50):
            try:
                resp = httpx.get("http://127.0.0.1:8003/status", timeout=1.0)
                if resp.status_code == 200:
                    break
            except Exception:
                time.sleep(0.1)
        else:
            pytest.fail("API server did not start")
        assert thread.is_alive()
        assert proc is dummy_proc
    finally:
        stop()
        thread.join(timeout=1)

    assert dummy_proc.terminated
