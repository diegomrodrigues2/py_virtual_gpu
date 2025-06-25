import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu import VirtualGPU, DevicePointer, GlobalMemory


def test_device_pointer_equality_and_repr():
    gm = GlobalMemory(32)
    a = DevicePointer(5, gm)
    b = DevicePointer(5, gm)
    c = DevicePointer(6, gm)
    assert a == b
    assert a != c
    text = repr(a)
    assert "offset=5" in text and "mem=GlobalMemory" in text


def test_gpu_malloc_free_and_reuse():
    gpu = VirtualGPU(0, 64)
    p1 = gpu.malloc(10)
    p2 = gpu.malloc(20)
    p3 = gpu.malloc(5)
    assert p1.offset < p2.offset < p3.offset
    gpu.free(p2)
    p4 = gpu.malloc(15)
    assert isinstance(p4, DevicePointer)
    assert p4.offset == p2.offset


def test_gpu_double_free_error_and_out_of_memory():
    gpu = VirtualGPU(0, 16)
    ptr = gpu.malloc(16)
    with pytest.raises(MemoryError):
        gpu.malloc(1)
    gpu.free(ptr)
    with pytest.raises(ValueError):
        gpu.free(ptr)


def test_gpu_coalescence_on_free():
    gpu = VirtualGPU(0, 32)
    a = gpu.malloc(10)
    b = gpu.malloc(6)
    gpu.free(a)
    gpu.free(b)
    c = gpu.malloc(16)
    assert c.offset == a.offset


def test_fragmentation_stress_keeps_total_free():
    gpu = VirtualGPU(0, 128)
    start_free = sum(sz for _, sz in gpu.global_memory._free_list)
    for i in range(1, 20):
        ptrs = [gpu.malloc(i) for _ in range(3)]
        for p in ptrs:
            gpu.free(p)
    end_free = sum(sz for _, sz in gpu.global_memory._free_list)
    assert start_free == end_free


def test_memcpy_host_to_device_and_back():
    gpu = VirtualGPU(0, 32)
    host_buf = bytes(range(8))
    ptr = gpu.malloc(8)
    gpu.memcpy_host_to_device(host_buf, ptr)
    assert gpu.global_memory.read(ptr.offset, 8) == host_buf

    out = gpu.memcpy_device_to_host(ptr, 8)
    assert out == host_buf


def test_memcpy_validation_errors():
    gpu = VirtualGPU(0, 16)
    ptr = gpu.malloc(4)

    gpu.memcpy_host_to_device(b"\x00\x01", ptr)

    with pytest.raises(ValueError):
        gpu.memcpy_device_to_host(ptr, -1)

    gpu.free(ptr)
    with pytest.raises(ValueError):
        gpu.memcpy_host_to_device(b"abcd", ptr)

