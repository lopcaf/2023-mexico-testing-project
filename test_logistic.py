import pytest
from logistic import logistic_map
from numpy.testing import assert_allclose
import math

@pytest.mark.parametrize('inp, out' , [((0.1,2.2), 0.198), ((0.2,3.4), 0.544), ((0.75,1.7), 0.31875)])
def test_logistic_value(inp,out):
    """Double checks the value returned by logistic.py"""
    x = inp[0]
    r = inp[1]
    assert math.isclose(logistic_map(x,r),out)
