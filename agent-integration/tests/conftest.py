"""Shared fixtures and discovery for agent integration tests."""
import os
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "level1: C++ plugin source tests")
    config.addinivalue_line("markers", "level2: CLI harness tests")
    config.addinivalue_line("markers", "level3: headless processing tests")
    config.addinivalue_line("markers", "level4: GUI RPC tests")
    config.addinivalue_line("markers", "level5: MCP server tests")


def pytest_collection_modifyitems(config, items):
    """Automatically tag tests based on class name."""
    for item in items:
        for cls in (item.cls,):
            if cls is None:
                continue
            name = cls.__name__
            if "Level1" in name:
                item.add_marker(pytest.mark.level1)
            elif "Level2" in name:
                item.add_marker(pytest.mark.level2)
            elif "Level3" in name:
                item.add_marker(pytest.mark.level3)
            elif "Level4" in name:
                item.add_marker(pytest.mark.level4)
            elif "Level5" in name:
                item.add_marker(pytest.mark.level5)


def pytest_report_header(config):
    """Print discovery info so users know what will actually run."""
    from test_integration import REPO_ROOT, BUILD_DIR, BINARY_PATH, HAS_CLI
    lines = [
        f"ACV repo root : {REPO_ROOT}",
        f"ACV build dir : {BUILD_DIR} ({'exists' if BUILD_DIR.exists() else 'NOT FOUND'})",
        f"ACV binary    : {BINARY_PATH or 'NOT FOUND'}",
        f"CLI harness   : {'available' if HAS_CLI else 'not installed'}",
    ]
    return lines
