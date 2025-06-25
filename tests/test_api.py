import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu.api import app


def test_status_endpoint():
    with TestClient(app) as client:
        resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_sms"] == 1
        assert data["global_mem_size"] == 1024
        assert data["shared_mem_size"] == 0


def test_openapi_schema_includes_status():
    with TestClient(app) as client:
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "/status" in schema.get("paths", {})
