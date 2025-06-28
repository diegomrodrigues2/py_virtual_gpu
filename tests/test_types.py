import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu import Half, Float32, Float64


def test_basic_arithmetic_and_rounding():
    a = Half(1.5)
    b = Half(2.25)
    res = a + b
    assert isinstance(res, Half)
    expected = np.float16(np.float16(1.5) + np.float16(2.25))
    assert float(res) == float(expected)


def test_promotion_to_float32():
    a = Half(1.2)
    b = Float32(2.3)
    res = a * b
    assert isinstance(res, Float32)
    expected = np.float32(np.float16(1.2) * np.float32(2.3))
    assert float(res) == float(expected)


def test_promotion_to_float64():
    a = Float32(1.5)
    b = Float64(2.0)
    res = b / a
    assert isinstance(res, Float64)
    expected = np.float64(np.float64(2.0) / np.float32(1.5))
    assert float(res) == float(expected)


def test_conversion_helpers():
    a = Half(3.0)
    assert isinstance(a.to_float32(), Float32)
    assert isinstance(a.to_float64(), Float64)
