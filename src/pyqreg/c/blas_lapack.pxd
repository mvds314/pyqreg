# -----------------------------
# Wrappers implemented in pyx
# -----------------------------

cdef void mm_dot(double* A, double* B, double* C,
                 int M, int N, int K,
                 int LDA, int LDB, int LDC,
                 double ALPHA, double BETA,
                 int transposeA, int transposeB)

cdef void blas_axpy(double* X, double A, double* Y, int N) 

cdef void mv_dot(double* A, double* X, double* Y,
                 int N, int D, int LDA, int transposeA) 

cdef void lapack_cholesky_decomp(double* A, int N)
cdef void lapack_cholesky_solve(double* A, double* b, int N)
cdef void lapack_cholesky_inv(double* A, int N)
