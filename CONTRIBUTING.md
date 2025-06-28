# Contributing

## Development setup

1. Optionally create a virtual environment and install the package in editable mode with the optional dependencies defined in `[project.optional-dependencies]` of `pyproject.toml`:

```bash
pip install -e .[api]
```

2. Also install the packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

3. Run the test suite at the repository root:

```bash
pytest
```
