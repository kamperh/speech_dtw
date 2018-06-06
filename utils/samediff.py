#!/usr/bin/env python

"""
Functions for performing same-different evaluation.

Details are given in:
- M. A. Carlin, S. Thomas, A. Jansen, and H. Hermansky, "Rapid evaluation of
  speech representations for spoken term discovery," in Proc. Interspeech,
  2011.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2014, 2015, 2018
"""

from __future__ import division
from __future__ import print_function
from datetime import datetime
from scipy.spatial.distance import pdist
import argparse
import numpy as np
import sys


#-----------------------------------------------------------------------------#
#                              SAMEDIFF FUNCTIONS                             #
#-----------------------------------------------------------------------------#

def average_precision(pos_distances, neg_distances, show_plot=False):
    """
    Calculate average precision and precision-recall breakeven.

    Return the average precision and precision-recall breakeven calculated
    using the true positive distances `pos_distances` and true negative
    distances `neg_distances`.
    """
    distances = np.concatenate(
        [pos_distances, neg_distances]
        )
    matches = np.concatenate(
        [np.ones(len(pos_distances)), np.zeros(len(neg_distances))]
        )

    # Sort from shortest to longest distance
    sorted_i = np.argsort(distances)
    distances = distances[sorted_i]
    matches = matches[sorted_i]

    # Calculate precision
    precision = np.cumsum(matches)/np.arange(1, len(matches) + 1)

    # Calculate average precision: the multiplication with matches and division
    # by the number of positive examples is to not count precisions at the same
    # recall point multiple times.
    average_precision = np.sum(precision * matches) / len(pos_distances)

    # Calculate recall
    recall = np.cumsum(matches)/len(pos_distances)

    # More than one precision can be at a single recall point, take the max one
    for n in range(len(recall) - 2, -1, -1):
        precision[n] = max(precision[n], precision[n + 1])

    # Calculate precision-recall breakeven
    prb_i = np.argmin(np.abs(recall - precision))
    prb = (recall[prb_i] + precision[prb_i])/2.

    if show_plot:
        import matplotlib.pyplot as plt
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")

    return average_precision, prb


def mean_average_precision(distances, labels):
    """
    Calculate mean average precision and precision-recall breakeven.

    Returns
    -------
    mean_average_precision, mean_prb, ap_dict : float, float, dict
        The dict gives the per-type average precisions.
    """

    label_matches = generate_matches_array(labels)  # across all tokens

    ap_dict = {}
    prbs = []
    for target_type in sorted(set(labels)):
        if len(np.where(np.asarray(labels) == target_type)[0]) == 1:
            continue
        type_matches = generate_type_matches_array(labels, target_type)
        swtt_matches = np.bitwise_and(
            label_matches == True, type_matches == True
            ) # same word, target type
        dwtt_matches = np.bitwise_and(
            label_matches == False, type_matches == True
            ) # different word, target type
        ap, prb = average_precision(
            distances[swtt_matches], distances[dwtt_matches]
            )
        prbs.append(prb)
        ap_dict[target_type] = ap
    return np.mean(ap_dict.values()), np.mean(prbs), ap_dict


def generate_matches_array(labels):
    """
    Return an array of bool in the same order as the distances from
    `scipy.spatial.distance.pdist` indicating whether a distance is for
    matching or non-matching labels.
    """
    N = len(labels)
    matches = np.zeros(int(N*(N - 1)/2), dtype=np.bool)

    # For every distance, mark whether it is a true match or not
    cur_matches_i = 0
    for n in range(N - 1):
        cur_label = labels[n]
        matches[cur_matches_i:cur_matches_i + (N - n) - 1] = np.asarray(
            labels[n + 1:]
            ) == cur_label
        cur_matches_i += N - n - 1

    return matches


def generate_type_matches_array(labels, target_type):
    """
    Return an array of bool in the same order as the distances from
    `scipy.spatial.distance.pdist` indicating whether a distance is between a
    pair involving `target_type`.
    """
    N = len(labels)
    matches = np.zeros(int(N*(N - 1)/2), dtype=np.bool)

    # For every distance, mark whether it is a true match or not
    cur_matches_i = 0
    for n in range(N - 1):
        cur_label = labels[n]
        if cur_label == target_type:
            matches[cur_matches_i:cur_matches_i + (N - n) - 1] = True
        else:
            matches[cur_matches_i:cur_matches_i + (N - n) - 1] = np.asarray(
                labels[n + 1:]
                ) == target_type
        cur_matches_i += N - n - 1

    return matches


def fixed_dim(X, labels, metric="cosine", show_plot=False):
    """
    Return average precision and precision-recall breakeven calculated on
    fixed-dimensional set `X`.

    `X` contains the fixed-dimensional data items as row vectors.
    """
    N, D = X.shape
    matches = generate_matches_array(labels)
    distances = pdist(X, metric)
    return average_precision(
        distances[matches == True], distances[matches == False], show_plot
        )


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument("labels_fn", help="file of labels")
    parser.add_argument(
        "distances_fn",
        help="file providing the distances between each pair of labels in the "
        "same order as `scipy.spatial.distance.pdist`"
        )
    parser.add_argument(
        "--binary_dists", dest="binary_dists", action="store_true",
        help="distances are given in float32 binary format "
        "(default is to assume distances are given in text format)"
        )
    parser.set_defaults(binary_dists=False)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    
    args = check_argv()

    # Read labels
    labels = [i.strip() for i in open(args.labels_fn)]
    N = len(labels)

    # Read distances
    print("Start time: " + str(datetime.now()))
    if args.binary_dists:
        print("Reading distances from binary file:", args.distances_fn)
        distances_vec = np.fromfile(args.distances_fn, dtype=np.float32)
    else:
        print("Reading distances from text file:", args.distances_fn)
        distances_vec = np.fromfile(
            args.distances_fn, dtype=np.float32, sep="\n"
            )
    if np.isnan(np.sum(distances_vec)):
        print("Warning: Distances contain nan")

    # Calculate average precision
    print("Calculating statistics.")
    matches = generate_matches_array(labels)
    ap, prb = average_precision(
        distances_vec[matches == True], distances_vec[matches == False], False
        )
    print("Average precision:", ap)
    print("Precision-recall breakeven:", prb)
    print("End time: " + str(datetime.now()))


if __name__ == "__main__":
    main()
