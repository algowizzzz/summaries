import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--file_path", action="store", default=None, help="Path to the single file to test"
    )

@pytest.fixture
def file_path(request):
    return request.config.getoption("--file_path") 