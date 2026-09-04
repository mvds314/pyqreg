import numpy as np
import pytest
from pyqreg.c.stats import invnormal, normalden
from scipy.stats import norm


def test_invnormal():
    assert np.isclose(norm.ppf(0.95), invnormal(0.95))
    assert np.isclose(norm.ppf(0.1), invnormal(0.1))
    assert np.isclose(norm.ppf(0.999), invnormal(0.999))


@pytest.mark.parametrize(
    "p, expected",
    [
        (0.0, -np.inf),
        (1.0, np.inf),
        (-0.5, np.nan),
        (2.0, np.nan),
    ],
)
def test_invnormal_outside_the_open_unit_interval(p, expected):
    """Arguments outside (0, 1) must not return uninitialized memory.

    None of the three interpolation branches assigns x for these inputs, so the
    return value is whatever happened to be on the stack. cluster_cov reaches
    this through invnormal(theta +- h_nG) at extreme quantiles.
    """
    assert np.isnan(invnormal(p)) if np.isnan(expected) else invnormal(p) == expected


def test_normalden():
    assert np.isclose(norm.pdf(0.95), normalden(0.95))
    assert np.isclose(norm.pdf(0.1), normalden(0.1))
    assert np.isclose(norm.pdf(0.999), normalden(0.999))
