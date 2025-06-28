"""Kernel function decorator and wrapper."""

from __future__ import annotations

from functools import wraps
from typing import Callable, Optional, Tuple

from .virtualgpu import VirtualGPU

GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


class KernelFunction:
    """Callable wrapper that dispatches execution to :class:`VirtualGPU`.

    When invoked, the wrapped function receives ``threadIdx``, ``blockIdx``,
    ``blockDim`` and ``gridDim`` as the first four positional arguments.
    """

    def __init__(
        self,
        func: Callable[..., object],
        grid_dim: Optional[GridDim],
        block_dim: Optional[BlockDim],
    ) -> None:
        wraps(func)(self)
        self._func = func
        self.grid_dim = grid_dim
        self.block_dim = block_dim

    def __call__(self, *args: object, **kwargs: object) -> object:
        gd = self.grid_dim or kwargs.pop("grid_dim", None)
        bd = self.block_dim or kwargs.pop("block_dim", None)
        cooperative = kwargs.pop("cooperative", False)
        if gd is None or bd is None:
            raise TypeError(
                f"Kernel '{self.__name__}' requires grid_dim and block_dim"
            )
        gpu = VirtualGPU.get_current()
        return gpu.launch_kernel(
            self._func,
            gd,
            bd,
            *args,
            cooperative=cooperative,
            **kwargs,
        )


def kernel(
    func: Optional[Callable[..., object]] = None,
    *,
    grid_dim: Optional[GridDim] = None,
    block_dim: Optional[BlockDim] = None,
) -> Callable[[Callable[..., object]], KernelFunction] | KernelFunction:
    """Decorate ``func`` turning it into a :class:`KernelFunction`."""

    if func is None:
        def wrapper(f: Callable[..., object]) -> KernelFunction:
            return KernelFunction(f, grid_dim, block_dim)

        return wrapper
    return KernelFunction(func, grid_dim, block_dim)


__all__ = ["kernel", "KernelFunction"]

