#!/usr/bin/env python

"""
Calculate DTW paths of a set of pairs from features in a given archive.

Author: Herman Kamper
Contact: h.kamper@sms.ed.ac.uk
Date: 2014
"""

from os import path
import argparse
import cPickle
import datetime
import numpy as np
import sys
import time

basedir = path.join(path.dirname(__file__), "../")
sys.path.append(basedir)

from kaldi import read_kaldi_ark
from speech_dtw import _dtw


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
    parser.add_argument(
        "paths_pkl_fn", help="a list of paths are pickeld to this file "
        "in the same order as which the pairs occur in `pairs_fn`; "
        "a single path is a list of tuples of frame pairs"
        )
    parser.add_argument(
        "--input_fmt", default="kaldi_txt", type=str, choices=["kaldi_txt", "npz"],
        help="the format of `features_fn` (default: %(default)s)"
        )    
    # parser.add_argument(
    #     "--binary_dists", dest="binary_dists", action="store_true",
    #     help="write distances in float32 binary format (default is not to do this)"
    #     )
    # parser.set_defaults(binary_dists=False)
    parser.add_argument(
        "--metric", default="cosine", type=str, choices=["cosine", "euclidean"],
        help="distance metric for calculating frame similarity for DTW (default: %(default)s)"
        )
    parser.add_argument(
        "--normalize_feats", dest="normalize_feats", action="store_true",
        help="normalize features per frame before calculating DTWs (default is not to normalize)"
        )
    parser.set_defaults(normalize_feats=False)
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
    paths_pkl_fn = args.paths_pkl_fn
    normalize_feats = args.normalize_feats

    n_paths_notify = 100

    # if args.metric == "cosine":
    #     dtw_cost_func = _dtw.multivariate_dtw_cost_cosine
    # elif args.metric == "euclidean":
    #     dtw_cost_func = _dtw.multivariate_dtw_cost_euclidean
        # normalize_feats = False

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
    print "Number of entries loaded:", len(ark)

    # sys.stdout.flush()

    # Normalize features per frame
    if normalize_feats:
        print "Normalizing features"
        for utt_id in ark:
            N = ark[utt_id].shape[0]
            for i in range(N):
                ark[utt_id][i, :] = ark[utt_id][i, :]/np.linalg.norm(ark[utt_id][i, :])

    # Calculate distances
    print "Calculating paths: ",
    paths = []
    i_paths = 0
    for i_pair, pair in enumerate(pairs):
        utt_id_1, utt_id_2 = pair
        paths.append(_dtw.multivariate_dtw(
            np.array(ark[utt_id_1], dtype=np.double), np.array(ark[utt_id_2], dtype=np.double)
            )[0])
        i_paths += 1
        if i_paths % n_paths_notify == 0:
            sys.stdout.write('.')
    print

    # costs = np.zeros(len(pairs))
    # for i_pair, pair in enumerate(pairs):
    #     utt_id_1, utt_id_2 = pair
    #     costs[i_pair] = dtw_cost_func(ark[utt_id_1], ark[utt_id_2], True)

    # Write to file
    print "Pickling paths:", paths_pkl_fn
    f = open(paths_pkl_fn, "wb")
    cPickle.dump(paths, f, -1)
    f.close()
    # if args.binary_dists:
    #     print "Writing distances to binary file:", distances_fn
    #     np.asarray(costs, dtype=np.float32).tofile(distances_fn)
    # else:
    #     print "Writing distances to text file:", distances_fn
    #     np.asarray(costs, dtype=np.float32).tofile(distances_fn, "\n")
    #     open(distances_fn, "a").write("\n")  # add final newline
    print "End time: " + str(datetime.datetime.now())


if __name__ == "__main__":
    main()
