import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu import VirtualGPU
from py_virtual_gpu.services import get_gpu_manager
from py_virtual_gpu.api.server import start_background_api
from py_virtual_gpu.kernel import kernel


@kernel(grid_dim=(1, 1, 1), block_dim=(2, 2, 1))
def mat_mul(threadIdx, blockIdx, blockDim, gridDim, a_ptr, b_ptr, c_ptr, n):
    col, row, _ = threadIdx
    acc = 0
    for k in range(n):
        a = int.from_bytes(a_ptr[row * n + k], "little", signed=True)
        b = int.from_bytes(b_ptr[k * n + col], "little", signed=True)
        acc += a * b
    c_ptr[row * n + col] = acc.to_bytes(4, "little", signed=True)


def host_mat_mul(A, B):
    n = len(A)
    C = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C


def main(with_api: bool = False) -> None:
    if with_api:
        api_thread, stop_api = start_background_api()
    else:
        api_thread = stop_api = None

    n = 2
    gpu = VirtualGPU(num_sms=1, global_mem_size=128)
    get_gpu_manager().add_gpu(gpu)
    VirtualGPU.set_current(gpu)

    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    expected = host_mat_mul(A, B)
    flat_A = [v for row in A for v in row]
    flat_B = [v for row in B for v in row]
    a_bytes = b"".join(v.to_bytes(4, "little", signed=True) for v in flat_A)
    b_bytes = b"".join(v.to_bytes(4, "little", signed=True) for v in flat_B)

    a_ptr = gpu.malloc(len(a_bytes))
    b_ptr = gpu.malloc(len(b_bytes))
    c_ptr = gpu.malloc(len(a_bytes))

    gpu.memcpy_host_to_device(a_bytes, a_ptr)
    gpu.memcpy_host_to_device(b_bytes, b_ptr)

    mat_mul(a_ptr, b_ptr, c_ptr, n)
    gpu.synchronize()

    out = gpu.memcpy_device_to_host(c_ptr, len(a_bytes))
    result = [int.from_bytes(out[i * 4:(i + 1) * 4], "little", signed=True) for i in range(n * n)]
    expected_flat = [v for row in expected for v in row]

    print("Kernel result:", result)
    print("Host result:", expected_flat)

    if stop_api:
        stop_api()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Matrix multiplication example")
    parser.add_argument("--api", action="store_true", help="start API server while running")
    args = parser.parse_args()
    main(with_api=args.api)
