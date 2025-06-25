import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu import VirtualGPU
from py_virtual_gpu.kernel import kernel


@kernel(grid_dim=(1, 1, 1), block_dim=(4, 1, 1))
def vec_mul(threadIdx, blockIdx, blockDim, gridDim, a_ptr, b_ptr, out_ptr):
    i = threadIdx[0]
    a = int.from_bytes(a_ptr[i], "little", signed=True)
    b = int.from_bytes(b_ptr[i], "little", signed=True)
    out_ptr[i] = (a * b).to_bytes(4, "little", signed=True)


def main() -> None:
    gpu = VirtualGPU(num_sms=0, global_mem_size=64)
    VirtualGPU.set_current(gpu)

    a_vals = [1, 2, 3, 4]
    b_vals = [5, 6, 7, 8]
    a_bytes = b"".join(v.to_bytes(4, "little", signed=True) for v in a_vals)
    b_bytes = b"".join(v.to_bytes(4, "little", signed=True) for v in b_vals)

    a_ptr = gpu.malloc(len(a_bytes))
    b_ptr = gpu.malloc(len(b_bytes))
    out_ptr = gpu.malloc(len(a_bytes))

    gpu.memcpy_host_to_device(a_bytes, a_ptr)
    gpu.memcpy_host_to_device(b_bytes, b_ptr)

    vec_mul(a_ptr, b_ptr, out_ptr)
    gpu.synchronize()

    out = gpu.memcpy_device_to_host(out_ptr, len(a_bytes))
    result = [int.from_bytes(out[i * 4:(i + 1) * 4], "little", signed=True) for i in range(4)]
    expected = [a_vals[i] * b_vals[i] for i in range(4)]

    print("Kernel result:", result)
    print("Host result:", expected)
    if result == expected:
        print("Vector multiplication successful!")
    else:
        print("Mismatch detected")


if __name__ == "__main__":
    main()
