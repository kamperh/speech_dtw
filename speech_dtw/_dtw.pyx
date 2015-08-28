"""
Dynamic time warping cost calculation functions.

Both paths and costs are calculated. Some of the distance calculation code is
based on:

- http://jakevdp.github.io/blog/2012/08/08/memoryview-benchmarks/

The dynamic programming code is based on:

- http://en.wikipedia.org/wiki/Dynamic_time_warping
- http://www.ee.columbia.edu/ln/labrosa/matlab/dtw/dp.m
- https://github.com/mdeklerk/DTW/blob/master/_dtw.pyx

The notation in the first reference was followed, while Dan Ellis's code
(second reference) was used to check for correctness.

Author: Herman Kamper
Contact: h.kamper@sms.ed.ac.uk
Date: 2014
"""

cimport cython
cimport numpy as np
from cpython cimport bool
from libc.math cimport sqrt
import numpy as np

cdef extern from "float.h":
    double DBL_MAX

# Define a function pointer to a metric function
ctypedef double (*metric_ptr)(
    double[:, ::1], double[:, ::1], Py_ssize_t, Py_ssize_t
    )


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double min3(double[3] v):
    cdef Py_ssize_t i, m = 0
    for i in range(1, 3):
        if v[i] < v[m]:
            m = i
    return v[m]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline Py_ssize_t i_min3(double[3] v):
    cdef Py_ssize_t i, m = 0
    for i in range(1, 3):
        if v[i] < v[m]:
            m = i
    return m


@cython.boundscheck(False)
@cython.wraparound(False)
def dp_cost(double[:, ::1] dist_mat):
    """
    Calculate the cost of the minimum-cost path through matrix `dist_mat`.

    The `dist_mat` can be calculated with `scipy.spatial.distance.cdist`. Only
    the overall cost is returned.
    """
    cdef int N, M
    cdef Py_ssize_t i, j
    cdef double[:, ::1] cost_mat
    cdef double[3] costs

    N = dist_mat.shape[0]
    M = dist_mat.shape[1]
    
    # Initialize the cost matrix
    cost_mat = np.zeros((N + 1, M + 1), dtype=np.double) + DBL_MAX
    cost_mat[0, 0] = 0.

    # Fill the cost matrix
    for i in range(N):
        for j in range(M):
            costs[0] = cost_mat[i, j]       # match (0)
            costs[1] = cost_mat[i, j + 1]   # insertion (1)
            costs[2] = cost_mat[i + 1, j]   # deletion (2)
            cost_mat[i + 1, j + 1] = dist_mat[i, j] + min3(costs)

    return cost_mat[N, M]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double cosine_dist(
        double[:, ::1] x, double[:, ::1] y, Py_ssize_t x_i, Py_ssize_t y_i
        ):
    """Calculate the cosine distance between `x[x_i, :]` and `y[y_i, :]`."""
    cdef int N = x.shape[1]
    cdef Py_ssize_t i
    cdef double dot = 0.
    cdef double norm_x = 0.
    cdef double norm_y = 0.
    for i in range(N):
        dot += x[x_i, i] * y[y_i, i]
        norm_x += x[x_i, i]*x[x_i, i]
        norm_y += y[y_i, i]*y[y_i, i]
    return 1. - dot/(sqrt(norm_x) * sqrt(norm_y))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double euclidean_dist(
        double[:, ::1] x, double[:, ::1] y, Py_ssize_t x_i, Py_ssize_t y_i
        ):
    """Calculate the Euclidean distance between `x[x_i, :]` and `y[y_i, :]`."""
    cdef int N = x.shape[1]
    cdef Py_ssize_t i
    cdef double sum_square_diffs = 0.
    for i in range(N):
        sum_square_diffs += (x[x_i, i] - y[y_i, i]) * (x[x_i, i] - y[y_i, i])
    return sqrt(sum_square_diffs)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double euclidean_squared_dist(
        double[:, ::1] x, double[:, ::1] y, Py_ssize_t x_i, Py_ssize_t y_i
        ):
    """Calculate the Euclidean distance between `x[x_i, :]` and `y[y_i, :]`."""
    cdef int N = x.shape[1]
    cdef Py_ssize_t i
    cdef double sum_square_diffs = 0.
    for i in range(N):
        sum_square_diffs += (x[x_i, i] - y[y_i, i]) * (x[x_i, i] - y[y_i, i])
    return sum_square_diffs


@cython.boundscheck(False)
@cython.wraparound(False)
def multivariate_dtw_cost(
        double[:, ::1] s, double[:, ::1] t, str metric="cosine"
        ):
    """
    Calculate the DTW alignment cost between vector time series `s` and `t`.

    The output of this function should be the same as calculating `dist_mat`
    using `scipy.spatial.distance.cdist` and then calling `dp_cost`.
    """
    cdef int N, M
    cdef Py_ssize_t i, j
    cdef double[:, ::1] cost_mat
    cdef double[3] costs

    cdef metric_ptr dist_func
    if metric == "cosine":
        dist_func = &cosine_dist
    elif metric == "euclidean":
        dist_func = &euclidean_dist
    else:
        raise ValueError("Unrecognized metric.")

    N = s.shape[0]
    M = t.shape[0]
    
    # Initialize the cost matrix
    cost_mat = np.zeros((N + 1, M + 1)) + DBL_MAX
    cost_mat[0, 0] = 0.

    # Fill the cost matrix
    for i in range(N):
        for j in range(M):
            costs[0] = cost_mat[i, j]       # match (0)
            costs[1] = cost_mat[i, j + 1]   # insertion (1)
            costs[2] = cost_mat[i + 1, j]   # deletion (2)
            cost_mat[i + 1, j + 1] = dist_func(s, t, i, j) + min3(costs)

    return cost_mat[N, M]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def multivariate_dtw_cost_cosine(
        double[:, ::1] s, double[:, ::1] t, bool dur_normalize=False
        ):
    """
    Calculate the DTW alignment cost between vector time series `s` and `t`
    using cosine distance.

    This function is the same as `multivariate_dtw_cost` but always uses cosine
    distance to help with speed (i.e. don't have to test metric distance type).
    """
    cdef int N, M
    cdef Py_ssize_t i, j
    cdef double[:, ::1] cost_mat
    cdef double[3] costs

    N = s.shape[0]
    M = t.shape[0]
    
    # Initialize the cost matrix
    cost_mat = np.zeros((N + 1, M + 1)) + DBL_MAX
    cost_mat[0, 0] = 0.

    # Fill the cost matrix
    for i in range(N):
        for j in range(M):
            costs[0] = cost_mat[i, j]       # match (0)
            costs[1] = cost_mat[i, j + 1]   # insertion (1)
            costs[2] = cost_mat[i + 1, j]   # deletion (2)
            cost_mat[i + 1, j + 1] = cosine_dist(s, t, i, j) + min3(costs)

    if dur_normalize:
        return cost_mat[N, M]/(N + M)
    else:
        return cost_mat[N, M]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def multivariate_dtw_cost_euclidean(
        double[:, ::1] s, double[:, ::1] t, bool dur_normalize=False
        ):
    """
    Calculate the DTW alignment cost between vector time series `s` and `t`
    using cosine distance.

    This function is the same as `multivariate_dtw_cost` but always uses
    euclidean distance to help with speed (i.e. don't have to test metric
    distance type).
    """
    cdef int N, M
    cdef Py_ssize_t i, j
    cdef double[:, ::1] cost_mat
    cdef double[3] costs

    N = s.shape[0]
    M = t.shape[0]
    
    # Initialize the cost matrix
    cost_mat = np.zeros((N + 1, M + 1)) + DBL_MAX
    cost_mat[0, 0] = 0.

    # Fill the cost matrix
    for i in range(N):
        for j in range(M):
            costs[0] = cost_mat[i, j]       # match (0)
            costs[1] = cost_mat[i, j + 1]   # insertion (1)
            costs[2] = cost_mat[i + 1, j]   # deletion (2)
            cost_mat[i + 1, j + 1] = euclidean_dist(s, t, i, j) + min3(costs)

    if dur_normalize:
        return cost_mat[N, M]/(N + M)
    else:
        return cost_mat[N, M]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def multivariate_dtw_cost_euclidean_squared(
        double[:, ::1] s, double[:, ::1] t, bool dur_normalize=False
        ):
    """
    Calculate the DTW alignment cost between vector time series `s` and `t`
    using cosine distance.

    This function is the same as `multivariate_dtw_cost` but always uses
    euclidean distance to help with speed (i.e. don't have to test metric
    distance type).
    """
    cdef int N, M
    cdef Py_ssize_t i, j
    cdef double[:, ::1] cost_mat
    cdef double[3] costs

    N = s.shape[0]
    M = t.shape[0]
    
    # Initialize the cost matrix
    cost_mat = np.zeros((N + 1, M + 1)) + DBL_MAX
    cost_mat[0, 0] = 0.

    # Fill the cost matrix
    for i in range(N):
        for j in range(M):
            costs[0] = cost_mat[i, j]       # match (0)
            costs[1] = cost_mat[i, j + 1]   # insertion (1)
            costs[2] = cost_mat[i + 1, j]   # deletion (2)
            cost_mat[i + 1, j + 1] = euclidean_squared_dist(s, t, i, j) + min3(costs)

    if dur_normalize:
        return cost_mat[N, M]/(N + M)
    else:
        return cost_mat[N, M]


@cython.boundscheck(False)
@cython.wraparound(False)
def multivariate_dtw(double[:, ::1] s, double[:, ::1] t, str metric="cosine"):
    """
    Calculate the DTW alignment between vector time series `s` and `t` and
    return the cost and path.

    Duration normalization is not performed.
    """
    cdef int N, M
    cdef Py_ssize_t i, j, i_penalty
    cdef double[:, ::1] cost_mat
    cdef double[3] costs

    cdef metric_ptr dist_func
    if metric == "cosine":
        dist_func = &cosine_dist
    elif metric == "euclidean":
        dist_func = &euclidean_dist
    else:
        raise ValueError("Unrecognized metric.")

    N = s.shape[0]
    M = t.shape[0]
    
    # Initialize the cost matrix
    cost_mat = np.zeros((N + 1, M + 1)) + DBL_MAX
    cost_mat[0, 0] = 0.

    # Fill the cost matrix
    traceback_mat = np.zeros((N, M), dtype=np.uint16)
    for i in range(N):
        for j in range(M):
            costs[0] = cost_mat[i, j]       # match (0)
            costs[1] = cost_mat[i, j + 1]   # insertion (1)
            costs[2] = cost_mat[i + 1, j]   # deletion (2)
            i_penalty = i_min3(costs)
            traceback_mat[i, j] = i_penalty
            cost_mat[i + 1, j + 1] = dist_func(s, t, i, j) + costs[i_penalty]

    # Trace back from bottom right
    i = N - 1
    j = M - 1
    cdef list path = [(i, j)]
    while i > 0 or j > 0:
        tb_type = traceback_mat[i, j]
        if tb_type == 0:
            # Match
            i = i - 1
            j = j - 1
        elif tb_type == 1:
            # Insertion
            i = i - 1
        elif tb_type == 2:
            # Deletion
            j = j - 1
        path.append((i, j))

    return (path, cost_mat[N, M])


# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# def multivariate_dtw_cosine(double[:, ::1] s, double[:, ::1] t, bool dur_normalize=False):
#     """
#     Calculate the DTW alignment between vector time series `s` and `t` using
#     cosine distance and return the cost and path.
#     """
#     cdef int N, M
#     cdef Py_ssize_t i, j
#     cdef double[:, ::1] cost_mat
#     cdef double[3] costs

#     N = s.shape[0]
#     M = t.shape[0]
    
#     # Initialize the cost matrix
#     cost_mat = np.zeros((N + 1, M + 1)) + DBL_MAX
#     cost_mat[0, 0] = 0.

#     # Fill the cost matrix
#     for i in range(N):
#         for j in range(M):
#             costs[0] = cost_mat[i, j]       # match (0)
#             costs[1] = cost_mat[i, j + 1]   # insertion (1)
#             costs[2] = cost_mat[i + 1, j]   # deletion (2)
#             cost_mat[i + 1, j + 1] = cosine_dist(s, t, i, j) + min3(costs)

#     if dur_normalize:
#         return cost_mat[N, M]/(N + M)
#     else:
#         return cost_mat[N, M]
