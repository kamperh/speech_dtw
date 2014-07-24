#!/usr/bin/env python

"""
Calculate DTW distances of a set of pairs from features in a Kaldi archive.

The distances are calculated in the same order as which the pairs occur in the 
pairs file.

Author: Herman Kamper
Contact: h.kamper@sms.ed.ac.uk
Date: 2014
"""

import argparse
import datetime
import numpy as np
import sys
import time

from kaldi import read_kaldi_ark
import _dtw_cost


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument(
        "pairs_fn", help="a file of a list of the pairs of utterance IDs for which distances should be "
        "calculated"
        )
    parser.add_argument(
        "features_fn", help="the file containing features; "
        "by default this should be a Kaldi archive in text format"
        )
    parser.add_argument("distances_fn", help="the distances are written to this file")
    parser.add_argument(
        "--input_fmt", default="kaldi_txt", type=str, choices=["kaldi_txt", "npz"],
        help="the format of `features_fn` (default: %(default)s)"
        )    
    parser.add_argument(
        "--binary_dists", dest="binary_dists", action="store_true",
        help="write distances in float32 binary format (default is not to do this)"
        )
    parser.set_defaults(binary_dists=False)
    parser.add_argument(
        "--metric", default="cosine", type=str, choices=["cosine", "euclidean"],
        help="distance metric for calculating frame similarity for DTW (default: %(default)s)"
        )
    parser.add_argument(
        "--no_normalize_feats", dest="normalize_feats", action="store_false",
        help="do not normalize features per frame before calculating DTWs (default is to normalize); if "
        "euclidean distance metric is used, normalization is not performed"
        )
    parser.set_defaults(normalize_feats=True)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def read_pairs(pairs_fn):
    """Return a list of tuples with pairs of utterance IDs."""
    pairs = []
    for line in open(pairs_fn):
        utt_1, utt_2 = line.split()
        pairs.append((utt_1, utt_2))
    return pairs


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():

    args = check_argv()
    pairs_fn = args.pairs_fn
    features_fn = args.features_fn
    distances_fn = args.distances_fn
    normalize_feats = args.normalize_feats

    if args.metric == "cosine":
        dtw_cost_func = _dtw_cost.multivariate_dtw_cost_cosine
    elif args.metric == "euclidean":
        dtw_cost_func = _dtw_cost.multivariate_dtw_cost_euclidean
        normalize_feats = False

    # Read the pairs and the archive
    print "Start time: " + str(datetime.datetime.now())
    print "Reading pairs from:", pairs_fn
    pairs = read_pairs(pairs_fn)
    print "Reading features from:", features_fn
    if args.input_fmt == "kaldi_txt":
        ark = read_kaldi_ark(features_fn)
    elif args.input_fmt == "npz":
        ark = np.load(features_fn)
        ark = dict(ark)

    # sys.stdout.flush()

    # Normalize features per frame
    if normalize_feats:
        print "Normalizing features."
        for utt_id in ark:
            N = ark[utt_id].shape[0]
            for i in range(N):
                ark[utt_id][i, :] = ark[utt_id][i, :]/np.linalg.norm(ark[utt_id][i, :])

    # Calculate distances
    print "Calculating distances."
    costs = np.zeros(len(pairs))
    for i_pair, pair in enumerate(pairs):
        utt_id_1, utt_id_2 = pair
        costs[i_pair] = dtw_cost_func(ark[utt_id_1], ark[utt_id_2], True)

    # Write to file
    if args.binary_dists:
        print "Writing distances to binary file:", distances_fn
        np.asarray(costs, dtype=np.float32).tofile(distances_fn)
    else:
        print "Writing distances to text file:", distances_fn
        np.asarray(costs, dtype=np.float32).tofile(distances_fn, "\n")
        open(distances_fn, "a").write("\n")  # add final newline
    print "End time: " + str(datetime.datetime.now())


if __name__ == "__main__":
    main()
