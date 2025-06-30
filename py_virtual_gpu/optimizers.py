"""Optimization kernels and helper functions."""

from __future__ import annotations

from typing import Tuple

from .kernel import kernel
from .types import Float32, Half, sqrt_numeric


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
    """Fused Adam update on ``n`` parameters.

    Each thread reads the gradient for one parameter, updates its first and
    second moment estimates (``m`` and ``v``), applies bias correction via
    ``corr1`` and ``corr2``, and writes back the new parameter value in a
    single kernel invocation.

    Parameters correspond to the Adam formula components: device pointers for
    parameters and gradients, moment buffers ``m`` and ``v``, the learning rate
    ``lr``, decay coefficients ``beta1`` and ``beta2``, numerical stability
    term ``eps``, and the precomputed correction factors ``corr1`` and
    ``corr2``.
    """
    i = blockIdx[0] * blockDim[0] + threadIdx[0]
    if i < n:
        g = grad_ptr[i]
        m = m_ptr[i].to_float32()
        v = v_ptr[i].to_float32()
        p = param_ptr[i]

        m_new = beta1 * m + (Float32(1.0) - beta1) * g
        v_new = beta2 * v + (Float32(1.0) - beta2) * (g * g)

        if m_ptr.dtype is Half:
            m_ptr[i] = Half(float(m_new))
        else:
            m_ptr[i] = m_new

        if v_ptr.dtype is Half:
            v_ptr[i] = Half(float(v_new))
        else:
            v_ptr[i] = v_new

        m_hat = m_new / corr1
        v_hat = v_new / corr2
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
