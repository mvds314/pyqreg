# blas_lapack.pxd
from cython cimport boundscheck, cdivision, wraparound
cimport numpy as np

ctypedef np.npy_float64 DOUBLE_t


cdef void _psi_function(
	double* a, 
	double* b,
	double theta, 
	int n
)
