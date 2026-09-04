import numpy as np
import pytest
import statsmodels.api as sm
from pyqreg import QuantReg


def make_data(n=600, seed=7):
    rng = np.random.default_rng(seed)
    X = np.asfortranarray(np.column_stack([np.ones(n), rng.normal(size=n)]))
    y = np.asfortranarray(X @ np.array([1.0, 2.0]) + rng.normal(size=n))
    return y, X


@pytest.mark.parametrize("kernel", ["biw", "cos", "epa", "gau", "par"])
def test_every_advertised_kernel_matches_statsmodels(kernel):
    """Each kernel named in the error message must produce statsmodels' bse.

    statsmodels is the reference implementation this covariance estimator was
    ported from, so it is an independent oracle.
    """
    y, X = make_data()

    bse = QuantReg(y, X).fit(0.5, cov_type="robust", kernel=kernel).bse
    expected = sm.QuantReg(y, X).fit(q=0.5, kernel=kernel, bandwidth="hsheather").bse

    np.testing.assert_allclose(bse, np.asarray(expected), rtol=1e-5)
