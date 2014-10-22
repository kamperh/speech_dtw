#!/usr/bin/env python

"""
Calculate the distances between fixed-dimensional vectors and write to file.

Author: Herman Kamper
Contact: h.kamper@sms.ed.ac.uk
Date: 2014
"""

from scipy.spatial.distance import pdist
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
        "features_fn", help="the file containing features; "
        "this should be in .npz numpy archive format"
        )
    parser.add_argument("labels_fn", help="output labels file")
    parser.add_argument(
        "distances_fn", help="the distances are written to this file"
        )
    parser.add_argument(
        "--binary_dists", dest="binary_dists", action="store_true",
        help="write distances in float32 binary format (default is not to do this)"
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
    print "Reading features from:", args.features_fn
    ark = np.load(args.features_fn)

    print "Calculating distances"
    X = []
    labels = []
    for label in ark:
        labels.append(label)
        X.append(ark[label])
    X = np.array(X)
    distances = pdist(X, metric="cosine")


    # print "Calculating distances"
    # distances = np.zeros(len(pairs))
    # for i_pair, pair in enumerate(pairs):
    #     utt_id_1, utt_id_2 = pair
    #     distances[i_pair] = cosine(ark[utt_id_1], ark[utt_id_2])

    # Write to file
    if args.binary_dists:
        print "Writing distances to binary file:", args.distances_fn
        np.asarray(distances, dtype=np.float32).tofile(args.distances_fn)
    else:
        print "Writing distances to text file:", args.distances_fn
        np.asarray(distances, dtype=np.float32).tofile(args.distances_fn, "\n")
        open(args.distances_fn, "a").write("\n")  # add final newline
    print "Writing labels to file:", args.labels_fn
    open(args.labels_fn, "w").write("\n".join(labels) + "\n")
    print datetime.datetime.now()


if __name__ == "__main__":
    main()
