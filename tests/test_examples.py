import os
import sys
import importlib
import ast
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _parse_results(output: str):
    kernel = host = None
    for line in output.splitlines():
        if line.startswith("Kernel result:"):
            kernel = ast.literal_eval(line.split(":", 1)[1].strip())
        if line.startswith("Host result:"):
            text = line.split(":", 1)[1].strip()
            if "np.float" in text:
                text = (
                    text.replace("np.float16(", "")
                    .replace("np.float32(", "")
                    .replace("np.float64(", "")
                    .replace(")", "")
                )
            host = ast.literal_eval(text)
    return kernel, host


def test_vector_mul_example(capsys):
    mod = importlib.import_module("examples.vector_mul")
    mod.main()
    kernel, host = _parse_results(capsys.readouterr().out)
    assert kernel == host


def test_convolution_example(capsys):
    mod = importlib.import_module("examples.convolution_2d")
    mod.main()
    kernel, host = _parse_results(capsys.readouterr().out)
    assert kernel == host


def test_matrix_mul_example(capsys):
    mod = importlib.import_module("examples.matrix_mul")
    mod.main()
    kernel, host = _parse_results(capsys.readouterr().out)
    assert kernel == host

def test_reduction_example(capsys):
    mod = importlib.import_module("examples.reduction_sum")
    mod.main()
    kernel, host = _parse_results(capsys.readouterr().out)
    assert kernel == host


def test_reduction_multi_example(capsys):
    mod = importlib.import_module("examples.reduction_sum_multi")
    mod.main()
    kernel, host = _parse_results(capsys.readouterr().out)
    assert kernel == host
    
def test_vector_sum_atomic_example(capsys):
    mod = importlib.import_module("examples.vector_sum_atomic")
    mod.main()
    kernel, host = _parse_results(capsys.readouterr().out)
    assert kernel == host


def test_mixed_precision_example(capsys):
    mod = importlib.import_module("examples.mixed_precision")
    mod.main()
    kernel, host = _parse_results(capsys.readouterr().out)
    assert pytest.approx(kernel, rel=1e-6) == host


def test_inspect_allocations_example(capsys):
    mod = importlib.import_module("examples.inspect_allocations")
    mod.main()
    kernel, host = _parse_results(capsys.readouterr().out)
    assert kernel == host


def test_adam_basic_example(capsys):
    from examples import adam_basic

    adam_basic.main()
    kernel, host = _parse_results(capsys.readouterr().out)
    assert kernel == pytest.approx(host)
