#!/usr/bin/env python

"""
Example on randomly generated input.

Author: Herman Kamper
Contact: h.kamper@sms.ed.ac.uk
Date: 2014
"""

from os import path
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as dist
import sys
import time

basedir = path.join(path.dirname(__file__), "../")
sys.path.append(basedir)

from speech_dtw import _dtw


#-----------------------------------------------------------------------------#
#                   PURE PYTHON DYNAMIC PROGRAMMING FUNCTION                  #
#-----------------------------------------------------------------------------#

def dp(dist_mat):
    """
    Find minimum-cost path through matrix `dist_mat` using dynamic programming.

    The cost of a path is defined as the sum of the matrix entries on that
    path. See the following for details of the algorithm:

    - http://en.wikipedia.org/wiki/Dynamic_time_warping
    - http://www.ee.columbia.edu/ln/labrosa/matlab/dtw/dp.m

    The notation in the first reference was followed, while Dan Ellis's code
    (second reference) was used to check for correctness. Return a list path 
    of indices and the cost matrix.
    """

    N, M = dist_mat.shape
    
    # Initialize the cost matrix
    cost_mat = np.zeros((N + 1, M + 1))
    for i in range(1, N + 1):
        cost_mat[i, 0] = np.inf
    for i in range(1, M + 1):
        cost_mat[0, i] = np.inf

    # Fill the cost matrix while keeping traceback information
    traceback_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            penalty = [
                cost_mat[i, j],      # match (0)
                cost_mat[i, j + 1],  # insertion (1)
                cost_mat[i + 1, j]]  # deletion (2)
            i_penalty = np.argmin(penalty)
            cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalty[i_penalty]
            traceback_mat[i, j] = i_penalty

    # Traceback from bottom right
    i = N - 1
    j = M - 1
    path = [(i, j)]
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

    # Strip infinity edges from cost_mat before returning
    cost_mat = cost_mat[1:, 1:]
    return (path[::-1], cost_mat)


#-----------------------------------------------------------------------------#
#                                    SCRIPT                                   #
#-----------------------------------------------------------------------------#

np.random.seed(1)

D = 13
s = np.random.rand(80, D)
t = np.random.rand(60, D)

# Use pure python code
start = time.time()
dist_mat = dist.cdist(s, t, "cosine")
path, cost_mat = dp(dist_mat)
print "Cost:", cost_mat[-1, -1]
elapsed = (time.time() - start)
print "Time:", elapsed*1000, "ms"

plt.subplot(131)
plt.title("Distance matrix")
plt.imshow(dist_mat, cmap=plt.cm.binary, interpolation="nearest")
plt.ylim([len(s), -1])
plt.xlim([-1, len(t)])
plt.subplot(132)
plt.title("Cost matrix")
plt.imshow(cost_mat, cmap=plt.cm.binary, interpolation="nearest")
t_path, s_path = zip(*path)
plt.plot(s_path, t_path)
plt.ylim([len(s), -1])
plt.xlim([-1, len(t)])

# Use cyton code
start = time.time()
dist_mat = dist.cdist(s, t, "cosine")
print "\nCost:", _dtw.dp_cost(dist_mat)
elapsed = time.time() - start
print "Time:", elapsed*1000, "ms"

start = time.time()
print "\nCost:", _dtw.multivariate_dtw_cost_cosine(s, t)
elapsed = time.time() - start
print "Time:", elapsed*1000, "ms"

start = time.time()
path2, cost = _dtw.multivariate_dtw(s, t)
print "\nCost:", cost
elapsed = time.time() - start
print "Time:", elapsed*1000, "ms"
plt.subplot(133)
plt.title("Cost matrix")
plt.imshow(cost_mat, cmap=plt.cm.binary, interpolation="nearest")
t_path, s_path = zip(*path2)
plt.plot(s_path, t_path)
plt.ylim([len(s), -1])
plt.xlim([-1, len(t)])

plt.show()
