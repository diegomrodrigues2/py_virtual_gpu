import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from py_virtual_gpu import (
    Half,
    Float32,
    Float64,
    sqrt_numeric,
    sin_numeric,
    cos_numeric,
    exp_numeric,
    log_numeric,
)


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


def test_sqrt_numeric():
    val = Float32(4.0)
    res = sqrt_numeric(val)
    assert isinstance(res, Float32)
    assert float(res) == float(np.sqrt(np.float32(4.0)))


def test_trigonometric_and_exp_log():
    val = Float32(np.pi / 2)
    sin_res = sin_numeric(val)
    assert isinstance(sin_res, Float32)
    assert np.isclose(float(sin_res), 1.0)

    cos_res = cos_numeric(Float32(0.0))
    assert isinstance(cos_res, Float32)
    assert np.isclose(float(cos_res), 1.0)

    exp_res = exp_numeric(Float32(0.0))
    assert isinstance(exp_res, Float32)
    assert np.isclose(float(exp_res), 1.0)

    log_res = log_numeric(Float32(1.0))
    assert isinstance(log_res, Float32)
    assert np.isclose(float(log_res), 0.0)

