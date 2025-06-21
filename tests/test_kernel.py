import pytest

from py_virtual_gpu.kernel import kernel, KernelFunction
from py_virtual_gpu.virtualgpu import VirtualGPU


class DummyGPU(VirtualGPU):
    def __init__(self):
        super().__init__(0, 0)
        self.calls = []

    def launch_kernel(self, func, grid_dim, block_dim, *args, **kwargs):
        self.calls.append((func, grid_dim, block_dim, args, kwargs))
        return "ok"


def test_decorator_without_params_and_explicit_dims():
    gpu = DummyGPU()
    VirtualGPU.set_current(gpu)

    @kernel
    def add(a, b):
        return a + b

    result = add(1, 2, grid_dim=(1, 1, 1), block_dim=(2, 2, 1))
    assert result == "ok"
    assert isinstance(add, KernelFunction)
    assert gpu.calls[0][0].__name__ == "add"
    assert gpu.calls[0][1] == (1, 1, 1)
    assert gpu.calls[0][2] == (2, 2, 1)
    assert gpu.calls[0][3] == (1, 2)


def test_decorator_with_params_uses_defaults():
    gpu = DummyGPU()
    VirtualGPU.set_current(gpu)

    @kernel(grid_dim=(4, 4, 1), block_dim=(8, 1, 1))
    def mul(a, b):
        return a * b

    result = mul(3, 4)
    assert result == "ok"
    assert gpu.calls[0][1] == (4, 4, 1)
    assert gpu.calls[0][2] == (8, 1, 1)


def test_missing_dimensions_raises_type_error():
    gpu = DummyGPU()
    VirtualGPU.set_current(gpu)

    @kernel
    def noop():
        pass

    with pytest.raises(TypeError):
        noop()


