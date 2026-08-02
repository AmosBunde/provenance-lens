"""Package skeleton tests: the distribution installs and every submodule imports."""

import importlib

import provenance_lens

SUBMODULES = ["data", "forensics", "baseline", "reasoner", "video", "eval", "demo"]


def test_version_is_exposed():
    assert provenance_lens.__version__ == "0.1.0"


def test_all_submodules_import():
    for name in SUBMODULES:
        module = importlib.import_module(f"provenance_lens.{name}")
        assert module.__doc__, f"submodule {name} is missing a docstring"
