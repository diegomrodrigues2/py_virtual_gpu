.PHONY: dev-api

dev-api:
python -m uvicorn py_virtual_gpu.api.main:app --reload --port 8000
