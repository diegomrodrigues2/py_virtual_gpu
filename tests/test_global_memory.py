import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu import GlobalMemory


def test_malloc_write_read():
    gm = GlobalMemory(32)
    ptr = gm.malloc(8)
    gm.write(ptr, b"abcdefgh")
    assert gm.read(ptr, 8) == b"abcdefgh"


def test_malloc_free_reuse():
    gm = GlobalMemory(16)
    p1 = gm.malloc(4)
    p2 = gm.malloc(4)
    gm.free(p1)
    p3 = gm.malloc(4)
    assert p3 == p1  # first-fit reuse
    assert p2 != p1


def test_memcpy_host_device_and_back():
    gm = GlobalMemory(16)
    ptr = gm.malloc(6)
    gm.memcpy(ptr, b"abcdef", 6, "HostToDevice")
    out = gm.memcpy(ptr, None, 6, "DeviceToHost")
    assert out == b"abcdef"


def test_memcpy_device_to_device():
    gm = GlobalMemory(16)
    src = gm.malloc(4)
    dest = gm.malloc(4)
    gm.write(src, b"1234")
    gm.memcpy(dest, src, 4, "DeviceToDevice")
    assert gm.read(dest, 4) == b"1234"

