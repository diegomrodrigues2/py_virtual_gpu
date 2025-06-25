# Example Scripts

This directory contains small demonstrations using the virtual GPU simulator.

## Running

Execute the scripts directly with Python:

```bash
python examples/vector_mul.py
python examples/convolution_2d.py
```

Each program allocates memory on the virtual device, launches a kernel and prints
whether the result computed on the device matches the host computation.
