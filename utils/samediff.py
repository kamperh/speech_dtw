#!/usr/bin/env python

"""
Functionality for performing same-different evaluation.

Details are given in:
- M. A. Carlin, S. Thomas, A. Jansen, and H. Hermansky, "Rapid evaluation of
  speech representations for spoken term discovery," in Proc. Interspeech,
  2011.

Author: Herman Kamper
Contact: h.kamper@sms.ed.ac.uk
Date: 2014
"""

from scipy.spatial.distance import pdist
import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import sys


#-----------------------------------------------------------------------------#
#                              SAMEDIFF FUNCTIONS                             #
#-----------------------------------------------------------------------------#

def average_precision(pos_distances, neg_distances, show_plot):
    """
    Calculate average precision and precision-recall breakeven.

    Return the average precision and precision-recall breakeven calculated
    using the true positive distances `pos_distances` and true negative
    distances `neg_distances`.
    """
    distances = np.concatenate([pos_distances, neg_distances])
    matches = np.concatenate([np.ones(len(pos_distances)), np.zeros(len(neg_distances))])

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
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")

    return average_precision, prb


def generate_matches_array(labels):
    """
    Return an array of bool in the same order as the distances from
    `scipy.spatial.distance.pdist` indicating whether a distance is for
    matching or non-matching labels.
    """
    N = len(labels)
    matches = np.zeros(N*(N - 1)/2, dtype=np.bool)

    # For every distance, mark whether it is a true match or not
    cur_matches_i = 0
    for n in range(N):
        cur_label = labels[n]
        matches[cur_matches_i:cur_matches_i + (N - n) - 1] = np.asarray(labels[n + 1:]) == cur_label
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
    return average_precision(distances[matches == True], distances[matches == False], show_plot)


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("labels_fn", help="file of labels")
    parser.add_argument(
        "distances_fn",
        help="file providing the distances between each pair of labels in the same order as "
        "`scipy.spatial.distance.pdist`"
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
    print "Start time: " + str(datetime.datetime.now())
    if args.binary_dists:
        print "Reading distances from binary file:", args.distances_fn
        distances_vec = np.fromfile(args.distances_fn, dtype=np.float32)
    else:
        print "Reading distances from text file:", args.distances_fn
        distances_vec = np.fromfile(args.distances_fn, dtype=np.float32, sep="\n")
    if np.isnan(np.sum(distances_vec)):
        print "Warning: Distances contain nan"
        # distances_vec = np.nan_to_num(distances_vec)
        # distances_vec[np.where(np.isnan(distances_vec))] = np.inf

    # print "Reading distances from:", args.distances_fn
    # distances_vec = np.zeros(N*(N - 1)/2)
    # i_dist_vec = 0
    # for line in open(args.distances_fn):
    #     distances_vec[i_dist_vec] = float(line.strip())
    #     i_dist_vec += 1

    # Calculate average precision
    print "Calculating statistics."
    matches = generate_matches_array(labels)
    ap, prb = average_precision(distances_vec[matches == True], distances_vec[matches == False], False)
    print "Average precision:", ap
    print "Precision-recall breakeven:", prb
    print "End time: " + str(datetime.datetime.now())


if __name__ == "__main__":
    main()
