import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from py_virtual_gpu import VirtualGPU, DevicePointer


def test_h2d_and_d2h_identity():
    gpu = VirtualGPU(0, 512)
    ptr = gpu.malloc(64)
    data = bytes(range(64))
    gpu.memcpy_host_to_device(data, ptr)
    out = gpu.memcpy_device_to_host(ptr, 64)
    assert out == data


def test_transfer_metrics():
    gpu = VirtualGPU(
        0,
        256,
        host_latency_cycles=5,
        host_bandwidth_bpc=64,
        device_latency_cycles=5,
        device_bandwidth_bpc=64,
    )
    ptr = gpu.malloc(128)
    gpu.memcpy_host_to_device(b"a" * 128, ptr)
    gpu.memcpy_device_to_host(ptr, 128)
    stats = gpu.get_transfer_stats()
    assert stats["transfers"] == 2
    assert stats["transfer_bytes"] == 256
    assert stats["transfer_cycles"] == 14  # 7 cycles each
