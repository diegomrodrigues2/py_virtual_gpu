import os
import sys
import struct
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu import VirtualGPU, Half, Float32, atomicAdd_float32
from py_virtual_gpu.services import get_gpu_manager
from py_virtual_gpu.api.server import start_background_api
from py_virtual_gpu.kernel import kernel


@kernel(grid_dim=(1, 1, 1), block_dim=(4, 1, 1))
def dot_half_fp32(threadIdx, blockIdx, blockDim, gridDim, a_ptr, b_ptr, out_ptr):
    i = threadIdx[0]
    a = a_ptr[i]
    b = b_ptr[i]
    prod = a.to_float32() * b.to_float32()
    atomicAdd_float32(out_ptr, float(prod))


def main(with_api: bool = False) -> None:
    if with_api:
        api_thread, stop_api = start_background_api()
    else:
        api_thread = stop_api = None

    gpu = VirtualGPU(num_sms=0, global_mem_size=64)
    get_gpu_manager().add_gpu(gpu)
    VirtualGPU.set_current(gpu)

    a_vals = [1.0, 2.0, 3.0, 4.0]
    b_vals = [5.0, 6.0, 7.0, 8.0]
    a_bytes = np.array(a_vals, dtype=np.float16).tobytes()
    b_bytes = np.array(b_vals, dtype=np.float16).tobytes()

    a_ptr = gpu.malloc(len(a_vals), dtype=Half)
    b_ptr = gpu.malloc(len(b_vals), dtype=Half)
    out_ptr = gpu.malloc_type(1, Float32)
    out_ptr[0] = Float32(0.0)

    gpu.memcpy_host_to_device(a_bytes, a_ptr)
    gpu.memcpy_host_to_device(b_bytes, b_ptr)

    dot_half_fp32(a_ptr, b_ptr, out_ptr)
    gpu.synchronize()

    out = gpu.memcpy_device_to_host(out_ptr, 4)
    result = struct.unpack("<f", out)[0]

    expected = np.float32(0.0)
    for a, b in zip(a_vals, b_vals):
        expected += np.float32(a) * np.float32(b)
    expected = float(expected)

    print("Kernel result:", result)
    print("Host result:", expected)
    if abs(result - expected) < 1e-5:
        print("Mixed precision dot product successful!")
    else:
        print("Mismatch detected")

    if stop_api:
        stop_api()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mixed precision example")
    parser.add_argument("--api", action="store_true", help="start API server while running")
    args = parser.parse_args()
    main(with_api=args.api)
