import os
import sys
import time
import httpx
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu.api.server import start_background_api


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
