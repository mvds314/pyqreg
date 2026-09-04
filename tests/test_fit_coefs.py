import numpy as np
from pyqreg.c.fit_coefs import fit_coefs


def test_fit_coefs():

    X = np.hstack([np.ones(1000).reshape([1000, 1]), np.arange(1000).reshape([1000, 1])]).astype(
        np.double
    )
    y = np.arange(3, 1003).astype(np.double)

    X = np.array(X, np.double, order="F", ndmin=1)
    y = np.array(y, np.double, order="F", ndmin=1)

    coefs = fit_coefs(X, y, 0.5, 1e-6)

    # The tolerance is deliberately looser than the default of np.isclose. This
    # is a degenerate problem for an interior point method: y is an exact linear
    # function of X, so every residual is zero, and cond(X'X) is about 1.3e6.
    # The recovered coefficients therefore depend on the BLAS implementation.
    # Accelerate on Apple Silicon lands about 4e-5 from the exact solution where
    # OpenBLAS reaches 1e-13, which is far below any statistically meaningful
    # difference but outside the 3e-5 default threshold of np.isclose.
    np.testing.assert_allclose(coefs, np.asarray([3.0, 1.0]), atol=1e-3)
