import numpy as np
import pytest
from scipy.stats import norm

from pyqreg.c.stats import invnormal, normalden
import pyqreg.c.stats as c_stats


def test_invnormal():
    assert np.isclose(norm.ppf(0.95), invnormal(0.95))
    assert np.isclose(norm.ppf(0.1), invnormal(0.1))
    assert np.isclose(norm.ppf(0.999), invnormal(0.999))


def test_normalden():
    assert np.isclose(norm.pdf(0.95), normalden(0.95))
    assert np.isclose(norm.pdf(0.1), normalden(0.1))
    assert np.isclose(norm.pdf(0.999), normalden(0.999))
