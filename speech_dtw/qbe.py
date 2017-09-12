#!/usr/bin/env python

"""
Functions for performing query-by-example (QbE) search using DTW sweeps.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2017
"""

from os import path
import numpy as np
import sys

basedir = path.join(path.dirname(__file__), "..")
sys.path.append(basedir)

from speech_dtw import _dtw

dtw_cost_func = _dtw.multivariate_dtw_cost_cosine


def dtw_sweep(query_seq, search_seq, n_step=3):
    """
    Return a list of the DTW costs as `query_seq` is swept across `search_seq`.

    Step size can be specified with `n_step`.
    """
    i_start = 0
    n_query = query_seq.shape[0]
    n_search = search_seq.shape[0]
    sweep_costs = []
    while i_start <= n_search - n_query or i_start == 0:
        sweep_costs.append(
            dtw_cost_func(query_seq, search_seq[i_start:i_start + n_query],
            True)
            )
        i_start += n_step
    return sweep_costs


def dtw_sweep_min(query_seq, search_seq, n_step=3):
    """
    Return the minimum DTW cost as `query_seq` is swept across `search_seq`.

    Step size can be specified with `n_step`.
    """
    i_start = 0
    n_query = query_seq.shape[0]
    n_search = search_seq.shape[0]
    min_cost = np.inf
    while i_start <= n_search - n_query or i_start == 0:
        cost = dtw_cost_func(
            query_seq, search_seq[i_start:i_start + n_query], True
            )
        i_start += n_step
        if cost < min_cost:
            min_cost = cost

    return min_cost


def parallel_dtw_sweep_min(query_list, search_list, n_step=3, n_cpus=1):
    """
    Calculate the minimum DTW cost for a list of queries and search sequences.

    A list of lists is returned. The order matches that of `query_list`, with
    each entry giving a list of costs of that query to each of the items in
    `search_list`.
    """
    from joblib import Parallel, delayed
    costs = Parallel(n_jobs=n_cpus)(delayed
        (dtw_sweep_min)(query_seq, search_seq) for query_seq in query_list for
        search_seq in search_list
        )
    n_search = len(search_list)
    return [
        costs[i*n_search:(i + 1)*n_search] for i in
        range(int(np.floor(len(costs)/n_search)))
        ]
