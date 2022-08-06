import pytest

@pytest.fixture(scope="class")
def base_fixtures():
    """base function to initialise test fixtures"""
    class BaseTestSettings:
        
        data = {}

    bst = BaseTestSettings()
    return bst
