import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu import VirtualGPU
from py_virtual_gpu.kernel import kernel


@kernel(grid_dim=(1, 1, 1), block_dim=(4, 1, 1))
def pointer_kernel(threadIdx, blockIdx, blockDim, gridDim, ptr):
    idx = threadIdx[0]
    # read using indexing
    val = int.from_bytes(ptr[idx], "little", signed=True)
    val *= 2
    # pointer arithmetic and write
    p = ptr + idx
    p[0] = val.to_bytes(4, "little", signed=True)
    # read again through p and update using indexing
    new_val = int.from_bytes(p[0], "little", signed=True)
    ptr[idx] = (new_val + 1).to_bytes(4, "little", signed=True)


def test_pointer_indexing_and_arithmetic():
    gpu = VirtualGPU(0, 64)
    VirtualGPU.set_current(gpu)
    ptr = gpu.malloc(16)

    host_vals = [1, 2, 3, 4]
    host_buf = b"".join(v.to_bytes(4, "little", signed=True) for v in host_vals)
    gpu.memcpy_host_to_device(host_buf, ptr)

    pointer_kernel(ptr)

    out = gpu.memcpy_device_to_host(ptr, 16)
    res = [int.from_bytes(out[i*4:(i+1)*4], "little", signed=True) for i in range(4)]
    assert res == [3, 5, 7, 9]
