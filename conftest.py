"""Pytest configuration shared by the whole suite.

Lives at the repo root so `pytest` works from anywhere: previously the tests
only ran from the project directory, because they relied on the current working
directory happening to be on sys.path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent

# Make `gestureflow` and `scripts` importable without an editable install, so a
# fresh clone can run the suite immediately.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Point Path.home() at a temp dir.

    commands.py resolves ~/.gestureflow/commands.yaml at import time via a
    module constant, so tests that touch config resolution must not depend on
    whether the developer running them happens to have a real one installed.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def pytest_collection_modifyitems(config, items):
    """Mark the property-based and subprocess tests as slow."""
    for item in items:
        if "test_properties" in item.nodeid or "test_js_parity" in item.nodeid:
            item.add_marker(pytest.mark.slow)
