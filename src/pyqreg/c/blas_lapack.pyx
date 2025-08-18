
import numpy as np
cimport numpy as np
cimport cython

# -----------------------------
# Matrix-matrix multiplication
# -----------------------------
cdef void mm_dot(double* A, double* B, double* C,
                 int M, int N, int K,
                 int LDA, int LDB, int LDC,
                 double ALPHA, double BETA,
                 int transposeA, int transposeB):
    # Wrap raw pointers as numpy arrays
    cdef np.ndarray[np.double_t, ndim=2] A_np = np.PyArray_SimpleNewFromData(2, [M, K], np.NPY_DOUBLE, <void*>A)
    cdef np.ndarray[np.double_t, ndim=2] B_np = np.PyArray_SimpleNewFromData(2, [K, N], np.NPY_DOUBLE, <void*>B)
    cdef np.ndarray[np.double_t, ndim=2] C_np = np.PyArray_SimpleNewFromData(2, [M, N], np.NPY_DOUBLE, <void*>C)

    if transposeA:
        A_mat = A_np.T
    else:
        A_mat = A_np
    if transposeB:
        B_mat = B_np.T
    else:
        B_mat = B_np

    # Perform multiplication
    np.copyto(C_np, ALPHA * np.dot(A_mat, B_mat) + BETA * C_np)

# -----------------------------
# Vector addition: Y += A*X
# -----------------------------
cdef void blas_axpy(double* X, double A, double* Y, int N):
    cdef np.ndarray[np.double_t, ndim=1] X_np = np.PyArray_SimpleNewFromData(1, [N], np.NPY_DOUBLE, <void*>X)
    cdef np.ndarray[np.double_t, ndim=1] Y_np = np.PyArray_SimpleNewFromData(1, [N], np.NPY_DOUBLE, <void*>Y)
    Y_np += A * X_np

# -----------------------------
# Matrix-vector multiplication
# -----------------------------
cdef void mv_dot(double* A, double* X, double* Y,
                 int N, int D, int LDA, int transposeA):
    cdef np.ndarray[np.double_t, ndim=2] A_np = np.PyArray_SimpleNewFromData(2, [N, D], np.NPY_DOUBLE, <void*>A)
    cdef np.ndarray[np.double_t, ndim=1] X_np = np.PyArray_SimpleNewFromData(1, [D], np.NPY_DOUBLE, <void*>X)
    cdef np.ndarray[np.double_t, ndim=1] Y_np = np.PyArray_SimpleNewFromData(1, [N], np.NPY_DOUBLE, <void*>Y)

    if transposeA:
        np.copyto(Y_np, np.dot(A_np.T, X_np))
    else:
        np.copyto(Y_np, np.dot(A_np, X_np))

# -----------------------------
# Cholesky wrappers
# -----------------------------
cdef void lapack_cholesky_decomp(double* A, int N):
    cdef np.ndarray[np.double_t, ndim=2] A_np = np.PyArray_SimpleNewFromData(2, [N, N], np.NPY_DOUBLE, <void*>A)
    chol = np.linalg.cholesky(A_np)
    np.copyto(A_np, chol)

cdef void lapack_cholesky_solve(double* A, double* b, int N):
    cdef np.ndarray[np.double_t, ndim=2] A_np = np.PyArray_SimpleNewFromData(2, [N, N], np.NPY_DOUBLE, <void*>A)
    cdef np.ndarray[np.double_t, ndim=1] b_np = np.PyArray_SimpleNewFromData(1, [N], np.NPY_DOUBLE, <void*>b)
    x = np.linalg.solve(A_np, b_np)
    np.copyto(b_np, x)

cdef void lapack_cholesky_inv(double* A, int N):
    cdef np.ndarray[np.double_t, ndim=2] A_np = np.PyArray_SimpleNewFromData(2, [N, N], np.NPY_DOUBLE, <void*>A)
    invA = np.linalg.inv(A_np)
    np.copyto(A_np, invA)

def lapack_cholesky_inv_py(np.ndarray[np.double_t, ndim=2] A):
    N = A.shape[0]
    lapack_cholesky_inv(<double*>A.data, N)

