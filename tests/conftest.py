"""Marker wiring shared by the task verifier and the CPU self-check suite."""

import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA device")
    config.addinivalue_line("markers", "heavy: trains models; minutes, not seconds")


def pytest_collection_modifyitems(config, items):
    """Skip CUDA-only checks when there is no CUDA, rather than failing them.

    The task itself requires a GPU and the scored run always has one. This exists so
    the same suite can be pointed at a machine without one and still report on every
    structural, schema and integrity check.
    """
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="no CUDA device available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
