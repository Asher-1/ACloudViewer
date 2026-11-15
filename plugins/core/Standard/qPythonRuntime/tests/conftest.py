# from https://icetutor.com/question/how-to-pass-arguments-in-pytest-by-command-line/
# https://stackoverflow.com/questions/40880259/how-to-pass-arguments-in-pytest-by-command-line

from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption("--acloudviewer_exe", action="store", help="Path to the ACloudViewer binary under test")


@pytest.fixture
def acloudviewer_exe(request):
    value = request.config.getoption("--acloudviewer_exe")
    if not value:
        pytest.fail("Specify the binary under test with --acloudviewer_exe", pytrace=False)
    binary = Path(value).resolve()
    if not binary.is_file():
        pytest.fail(f"The binary under test does not exist: {binary}", pytrace=False)
    return str(binary)
