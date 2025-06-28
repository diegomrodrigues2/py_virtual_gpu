import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu import VirtualGPU, Float32, Half


def test_malloc_with_dtype_and_access():
    gpu = VirtualGPU(0, 64)
    ptr = gpu.malloc(2, dtype=Float32)
    assert ptr.element_size == 4
    ptr[0] = Float32(1.5)
    ptr[1] = Float32(2.5)
    assert isinstance(ptr[0], Float32)
    assert isinstance(ptr[1], Float32)
    assert float(ptr[0]) == 1.5
    assert float(ptr[1]) == 2.5


def test_malloc_type_helper_and_bytes_write():
    gpu = VirtualGPU(0, 32)
    ptr = gpu.malloc_type(1, Half)
    val = np.array([3.0], dtype=np.float16).tobytes()
    ptr[0] = val
    assert isinstance(ptr[0], Half)
    assert float(ptr[0]) == 3.0

