import pytest
import warnings


@pytest.fixture(autouse=True)
def _suppress_nannyml_warnings() -> None:
    """
    Suppress NannyML chunk size warnings in tests.
    These warnings are expected with small test datasets; production batches are much larger.
    """
    warnings.filterwarnings('ignore', message='.*number of chunks is too low.*', category=UserWarning)
