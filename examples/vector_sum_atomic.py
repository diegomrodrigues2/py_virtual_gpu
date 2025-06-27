import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu import VirtualGPU, atomicAdd
from py_virtual_gpu.kernel import kernel
from py_virtual_gpu.services import get_gpu_manager
from py_virtual_gpu.api.server import start_background_api


@kernel(grid_dim=(1, 1, 1), block_dim=(8, 1, 1))
def vec_sum(threadIdx, blockIdx, blockDim, gridDim, data_ptr, sum_ptr):
    i = threadIdx[0]
    val = int.from_bytes(data_ptr[i], "little", signed=True)
    atomicAdd(sum_ptr, val)


def main(with_api: bool = False) -> None:
    if with_api:
        api_thread, stop_api = start_background_api()
    else:
        api_thread = stop_api = None

    gpu = VirtualGPU(num_sms=0, global_mem_size=64)
    get_gpu_manager().add_gpu(gpu)
    VirtualGPU.set_current(gpu)

    values = [1, 2, 3, 4, 5, 6, 7, 8]
    data_bytes = b"".join(v.to_bytes(4, "little", signed=True) for v in values)

    data_ptr = gpu.malloc(len(data_bytes))
    sum_ptr = gpu.malloc(4)
    sum_ptr[0] = (0).to_bytes(4, "little", signed=True)

    gpu.memcpy_host_to_device(data_bytes, data_ptr)

    vec_sum(data_ptr, sum_ptr)
    gpu.synchronize()

    out = gpu.memcpy_device_to_host(sum_ptr, 4)
    result = int.from_bytes(out, "little", signed=True)
    expected = sum(values)

    print("Kernel result:", result)
    print("Host result:", expected)
    if result == expected:
        print("Vector sum successful!")
    else:
        print("Mismatch detected")

    if stop_api:
        stop_api()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Atomic vector sum example")
    parser.add_argument("--api", action="store_true", help="start API server while running")
    args = parser.parse_args()
    main(with_api=args.api)
