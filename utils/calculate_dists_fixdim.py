#!/usr/bin/env python

"""
Calculate the distances between fixed-dimensional vectors and write to file.

Author: Herman Kamper
Contact: h.kamper@sms.ed.ac.uk
Date: 2014
"""

import argparse
import datetime
import sys
import numpy as np

from calculate_dtw_costs import read_pairs


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
        "this should be in .npz numpy archive format"
        )
    parser.add_argument(
        "distances_fn", help="the distances are written to this file "
        "in the same order as which the pairs occur in `pairs_fn`"
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()
    
    print datetime.datetime.now()
    print "Reading pairs from:", args.pairs_fn
    pairs = read_pairs(args.pairs_fn)
    print "Reading features from:", features_fn
    if args.input_fmt == "kaldi_txt":
        ark = read_kaldi_ark(features_fn)
    elif args.input_fmt == "npz":
        ark = np.load(features_fn)
        ark = dict(ark)


    sys.exit()


    ark = np.load(args.vectors_npz_fn)
    ark = dict(ark)
    labels = sorted(ark.keys())
    m = len(labels)
    print m
    print "Writing labels:", args.labels_fn
    open(args.labels_fn, "w").write("\n".join(labels) + "\n")

    # # Generate distances for all possible pairs
    # f = open(pairs_fn, "w")
    # for i in xrange(0, m - 1):
    #     for j in xrange(i + 1, m):
    #         f.write(labels[i] + " " + labels[j] + "\n")
    # f.close()
    # print "Wrote pairs to file:", pairs_fn




if __name__ == "__main__":
    main()
