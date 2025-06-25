import os
import sys
import importlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_vector_mul_example(capsys):
    mod = importlib.import_module("examples.vector_mul")
    mod.main()
    out = capsys.readouterr().out.lower()
    assert "successful" in out


def test_convolution_example(capsys):
    mod = importlib.import_module("examples.convolution_2d")
    mod.main()
    out = capsys.readouterr().out.lower()
    assert "successful" in out
