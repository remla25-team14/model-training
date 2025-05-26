"""
Metamorphic-tests placeholder
"""

import pytest

@pytest.mark.metamorphic
@pytest.mark.xfail(strict=False, reason="Metamorphic tests to be implemented in next iteration")
def test_metamorphic_placeholder():
    """Dummy placeholder – always xfail."""
    assert False, "placeholder"