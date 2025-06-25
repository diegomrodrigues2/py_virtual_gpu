import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu import VirtualGPU


def test_device_pointer_index_read_write():
    gpu = VirtualGPU(0, 32)
    VirtualGPU.set_current(gpu)
    ptr = gpu.malloc(8)

    ptr[0] = b"abcd"
    ptr[1] = b"efgh"

    assert ptr[0] == b"abcd"
    assert ptr[1] == b"efgh"
    assert gpu.global_memory.read(ptr.offset, 4) == b"abcd"
    assert gpu.global_memory.read(ptr.offset + 4, 4) == b"efgh"


def test_device_pointer_setitem_size_mismatch():
    gpu = VirtualGPU(0, 16)
    VirtualGPU.set_current(gpu)
    ptr = gpu.malloc(4)
    with pytest.raises(ValueError):
        ptr[0] = b"ab"
