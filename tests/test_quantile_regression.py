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


def test_unknown_fit_method_is_rejected():
    """An unrecognised fit_method must be reported, not silently skipped.

    Without validation the coefficient step is skipped entirely and the failure
    surfaces later as a confusing AttributeError on self.params.
    """
    y, X = make_data()

    with pytest.raises(ValueError, match="fit_method"):
        QuantReg(y, X).fit(0.5, fit_method="ipmm")


def test_preproc_ipm_solves_to_the_same_optimum_as_plain_ipm():
    """Preprocessing is an exact reformulation, not an approximation.

    Portnoy and Koenker's preprocessing discards observations that provably
    cannot be interpolated, so it must reach the same optimum as plain ipm.
    Passing the wrong duality gap through to the solver loosens the stopping
    rule and quietly returns a coarser solution.
    """
    y, X = make_data(n=3000, seed=11)
    expected = np.asarray(sm.QuantReg(y, X).fit(q=0.5).params)

    params = QuantReg(y, X).fit(0.5, fit_method="preproc-ipm", seed=1).params

    np.testing.assert_allclose(params, expected, atol=1e-5)
