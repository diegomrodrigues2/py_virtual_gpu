"""FastAPI application for py_virtual_gpu."""

from .main import app
from .server import start_background_api, start_background_dashboard
__all__ = ["app", "start_background_api", "start_background_dashboard"]
