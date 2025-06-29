import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu import VirtualGPU, Float32
from py_virtual_gpu.kernel import kernel
from py_virtual_gpu.services import get_gpu_manager
from py_virtual_gpu.api.server import start_background_dashboard


@kernel(grid_dim=(1, 1, 1), block_dim=(4, 1, 1))
def add_vec(threadIdx, blockIdx, blockDim, gridDim, a_ptr, b_ptr, out_ptr):
    i = threadIdx[0]
    out_ptr[i] = a_ptr[i] + b_ptr[i]


def main(with_dashboard: bool = False) -> None:
    if with_dashboard:
        api_thread, ui_proc, stop = start_background_dashboard()
    else:
        api_thread = ui_proc = stop = None

    gpu = VirtualGPU(num_sms=0, global_mem_size=64)
    get_gpu_manager().add_gpu(gpu)
    VirtualGPU.set_current(gpu)

    a_vals = [1.0, 2.0, 3.0, 4.0]
    b_vals = [5.0, 6.0, 7.0, 8.0]

    a_ptr = gpu.malloc(len(a_vals), dtype=Float32, label="A")
    b_ptr = gpu.malloc(len(b_vals), dtype=Float32, label="B")
    out_ptr = gpu.malloc(len(a_vals), dtype=Float32, label="OUT")

    for i, v in enumerate(a_vals):
        a_ptr[i] = Float32(v)
    for i, v in enumerate(b_vals):
        b_ptr[i] = Float32(v)

    add_vec(a_ptr, b_ptr, out_ptr)
    gpu.synchronize()

    result = [float(out_ptr[i]) for i in range(len(a_vals))]
    expected = [a_vals[i] + b_vals[i] for i in range(len(a_vals))]

    print("Kernel result:", result)
    print("Host result:", expected)

    if stop:
        stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect allocations example")
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="launch API server and dashboard while running",
    )
    args = parser.parse_args()
    main(with_dashboard=args.dashboard)
