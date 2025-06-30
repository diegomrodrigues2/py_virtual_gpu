"""Numeric data types used inside GPU kernels."""

from __future__ import annotations

import operator
from typing import Any, Callable, Union

import numpy as np


Number = Union[int, float, np.number]


def _unwrap(value: Any) -> Any:
    if isinstance(value, (Half, Float32, Float64)):
        return value.value
    return value


def _wrap(value: Any, dtype: np.dtype) -> "Numeric":
    if dtype == np.float16:
        return Half(np.float16(value))
    if dtype == np.float32:
        return Float32(np.float32(value))
    if dtype == np.float64:
        return Float64(np.float64(value))
    raise TypeError(f"Unsupported dtype: {dtype}")


def _promote(a: Any, b: Any) -> np.dtype:
    return np.result_type(_unwrap(a), _unwrap(b))


class Numeric:
    dtype: Any
    value: Any

    def __init__(self, value: Number) -> None:
        self.value = np.dtype(self.dtype).type(value)

    # ------------------------------------------------------------------
    def _binary_op(self, other: Any, op: Callable[[Any, Any], Any], *, reverse: bool = False) -> "Numeric":
        dtype = _promote(self, other)
        if reverse:
            a_val = np.dtype(dtype).type(_unwrap(other))
            b_val = np.dtype(dtype).type(_unwrap(self))
        else:
            a_val = np.dtype(dtype).type(_unwrap(self))
            b_val = np.dtype(dtype).type(_unwrap(other))
        result = np.dtype(dtype).type(op(a_val, b_val))
        return _wrap(result, dtype)

    # Arithmetic ---------------------------------------------------------
    def __add__(self, other: Any) -> "Numeric":
        return self._binary_op(other, operator.add)

    def __radd__(self, other: Any) -> "Numeric":
        return self._binary_op(other, operator.add)

    def __sub__(self, other: Any) -> "Numeric":
        return self._binary_op(other, operator.sub)

    def __rsub__(self, other: Any) -> "Numeric":
        return self._binary_op(other, operator.sub, reverse=True)

    def __mul__(self, other: Any) -> "Numeric":
        return self._binary_op(other, operator.mul)

    def __rmul__(self, other: Any) -> "Numeric":
        return self._binary_op(other, operator.mul)

    def __truediv__(self, other: Any) -> "Numeric":
        return self._binary_op(other, operator.truediv)

    def __rtruediv__(self, other: Any) -> "Numeric":
        return self._binary_op(other, operator.truediv, reverse=True)

    # Comparison --------------------------------------------------------
    def __eq__(self, other: Any) -> bool:  # type: ignore[override]
        return float(self) == float(_unwrap(other))

    def __lt__(self, other: Any) -> bool:
        return float(self) < float(_unwrap(other))

    def __le__(self, other: Any) -> bool:
        return float(self) <= float(_unwrap(other))

    def __gt__(self, other: Any) -> bool:
        return float(self) > float(_unwrap(other))

    def __ge__(self, other: Any) -> bool:
        return float(self) >= float(_unwrap(other))

    # Conversion --------------------------------------------------------
    def to_float32(self) -> "Float32":
        return Float32(np.float32(self.value))

    def to_float64(self) -> "Float64":
        return Float64(np.float64(self.value))

    def __float__(self) -> float:
        return float(self.value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value})"


class Half(Numeric):
    """Half-precision floating point number (fp16)."""

    dtype = np.float16


class Float32(Numeric):
    """Single-precision floating point number (fp32)."""

    dtype = np.float32


class Float64(Numeric):
    """Double-precision floating point number (fp64)."""

    dtype = np.float64


def sqrt_numeric(value: Numeric) -> Numeric:
    """Return the square root of ``value`` preserving its dtype."""

    result = np.sqrt(_unwrap(value))
    return _wrap(result, np.dtype(value.dtype))


def sin_numeric(value: Numeric) -> Numeric:
    """Return ``sin(value)`` preserving its dtype."""

    result = np.sin(_unwrap(value))
    return _wrap(result, np.dtype(value.dtype))


def cos_numeric(value: Numeric) -> Numeric:
    """Return ``cos(value)`` preserving its dtype."""

    result = np.cos(_unwrap(value))
    return _wrap(result, np.dtype(value.dtype))


def exp_numeric(value: Numeric) -> Numeric:
    """Return ``exp(value)`` preserving its dtype."""

    result = np.exp(_unwrap(value))
    return _wrap(result, np.dtype(value.dtype))


def log_numeric(value: Numeric) -> Numeric:
    """Return ``log(value)`` preserving its dtype."""

    result = np.log(_unwrap(value))
    return _wrap(result, np.dtype(value.dtype))


__all__ = [
    "Half",
    "Float32",
    "Float64",
    "sqrt_numeric",
    "sin_numeric",
    "cos_numeric",
    "exp_numeric",
    "log_numeric",
]
