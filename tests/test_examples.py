import os
import sys
import importlib
import ast

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _parse_results(output: str):
    kernel = host = None
    for line in output.splitlines():
        if line.startswith("Kernel result:"):
            kernel = ast.literal_eval(line.split(":", 1)[1].strip())
        if line.startswith("Host result:"):
            host = ast.literal_eval(line.split(":", 1)[1].strip())
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
    
def test_vector_sum_atomic_example(capsys):
    mod = importlib.import_module("examples.vector_sum_atomic")
    mod.main()
    kernel, host = _parse_results(capsys.readouterr().out)
    assert kernel == host
