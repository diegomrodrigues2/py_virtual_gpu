"""Optimization kernels and helper functions."""

from __future__ import annotations

from typing import Tuple

from .kernel import kernel
from .types import Float32, sqrt_numeric


@kernel
def _adam_step_kernel(
    threadIdx,
    blockIdx,
    blockDim,
    gridDim,
    param_ptr,
    grad_ptr,
    m_ptr,
    v_ptr,
    lr,
    beta1,
    beta2,
    eps,
    corr1,
    corr2,
    n,
):
    i = blockIdx[0] * blockDim[0] + threadIdx[0]
    if i < n:
        g = grad_ptr[i]
        m = m_ptr[i]
        v = v_ptr[i]
        p = param_ptr[i]

        m = beta1 * m + (Float32(1.0) - beta1) * g
        v = beta2 * v + (Float32(1.0) - beta2) * (g * g)

        m_ptr[i] = m
        v_ptr[i] = v

        m_hat = m / corr1
        v_hat = v / corr2
        denom = sqrt_numeric(v_hat) + eps
        param_ptr[i] = p - lr * (m_hat / denom)


def adam_step(
    param_ptr,
    grad_ptr,
    m_ptr,
    v_ptr,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    t: int,
    n: int,
    *,
    grid_dim: Tuple[int, int, int] = (1, 1, 1),
    block_dim: Tuple[int, int, int] | None = None,
) -> None:
    """Perform one step of the Adam optimizer on device memory."""

    corr1 = Float32(1.0 - beta1.value ** t)
    corr2 = Float32(1.0 - beta2.value ** t)
    if block_dim is None:
        block_dim = (n, 1, 1)

    _adam_step_kernel(
        param_ptr,
        grad_ptr,
        m_ptr,
        v_ptr,
        lr,
        beta1,
        beta2,
        eps,
        corr1,
        corr2,
        n,
        grid_dim=grid_dim,
        block_dim=block_dim,
    )


__all__ = ["adam_step"]
