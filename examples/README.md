# Example Scripts

This directory contains small demonstrations using the virtual GPU simulator.

## Running

Execute the scripts directly with Python:

```bash
python examples/vector_mul.py
python examples/convolution_2d.py
python examples/matrix_mul.py
python examples/reduction_sum.py
python examples/reduction_sum_multi.py
python examples/mixed_precision.py
python examples/adam_basic.py
python examples/linear_regression_mse.py
python examples/logistic_regression.py
python examples/inspect_allocations.py
```

Each program allocates memory on the virtual device, launches a kernel and prints
whether the result computed on the device matches the host computation.

The `adam_basic.py` script demonstrates a few Adam optimizer steps
with bias correction enabled. It runs Adam with FP16 state variables
(`m` and `v`) while parameters remain in `Float32`.

### Visualizing via API and UI

All examples accept the ``--api`` flag to start the FastAPI server while they
run. The ``inspect_allocations.py`` script also provides ``--dashboard`` to
launch the UI automatically. When the server is active you can manually start
the dashboard from the ``app`` directory:

```bash
cd app && npm install && npm run dev
```

Then run an example with API support enabled:

```bash
python examples/vector_mul.py --api
python examples/matrix_mul.py --api
python examples/mixed_precision.py --api
python examples/inspect_allocations.py --dashboard
```

Open ``http://localhost:5173`` to inspect the execution step by step through the
UI. When the script finishes the API server shuts down automatically.

If you want to double check the data the UI is receiving you can hit the API
directly while an example is running:

```bash
curl http://localhost:8000/gpus
curl http://localhost:8000/gpus/0/state
curl http://localhost:8000/events
```
