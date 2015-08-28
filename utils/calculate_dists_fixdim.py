#!/usr/bin/env python

"""
Calculate the distances between fixed-dimensional vectors and write to file.

Author: Herman Kamper
Contact: h.kamper@sms.ed.ac.uk
Date: 2014-2015
"""

from scipy.spatial.distance import pdist
import argparse
import datetime
import sys
import scipy.linalg
import numpy as np

blas_norm, = scipy.linalg.get_blas_funcs(('nrm2',), (np.array([], dtype=np.float32),))


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
    parser.add_argument(
        "--normalize", dest="normalize", action="store_true",
        help="normalize embeddings to unit sphere before calculating distances (default is not to do this)"
        )
    parser.add_argument(
        "--metric", choices=["cosine", "euclidean", "hamming", "chebyshev"], default="cosine",
        help="distance metric (default: %(default)s)"
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

    if args.normalize:
        print "Normalizing embeddings"
    else:
        print "Reading embeddings"
    n_embeds = 0
    n_zeros = 0
    X = []
    labels = []
    for label in sorted(ark):
        labels.append(label)
        cur_embed = ark[label]
        if args.normalize:
            if blas_norm(cur_embed) != 0:
                X.append(cur_embed/blas_norm(cur_embed))
            else:
                X.append(np.zeros(cur_embed.shape))
                # n_zeros += 1
        else:
            X.append(cur_embed)
        n_embeds += 1
    X = np.array(X)
    print "No. embeddings:", n_embeds
    # print "No. all-zero embeddings:", n_zeros

    print "Calculating distances"
    distances = pdist(X, metric=args.metric)

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
