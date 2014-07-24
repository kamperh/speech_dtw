#!/usr/bin/env python

"""
Given a list of labels, generate a file with all the possible label pairs.

The order of the pairs is the same as that used when calculating distances
using `scipy.spatial.distance.pdist`.

Author: Herman Kamper
Contact: h.kamper@sms.ed.ac.uk
Date: 2014
"""

import argparse
import sys


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("labels_fn", help="a file of labels")
    parser.add_argument(
        "pairs_fn", help="output file with all possible label pairs, "
        "in the same order as `scipy.spatial.distance.pdist`"
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
    labels_fn = args.labels_fn
    pairs_fn = args.pairs_fn

    # Read list of IDs
    labels = [i.strip() for i in open(labels_fn)]
    m = len(labels)

    # Generate all possible pairs
    f = open(pairs_fn, "w")
    for i in xrange(0, m - 1):
        for j in xrange(i + 1, m):
            f.write(labels[i] + " " + labels[j] + "\n")
    f.close()
    print "Wrote pairs to file:", pairs_fn


if __name__ == "__main__":
    main()
