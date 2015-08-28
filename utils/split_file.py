#!/usr/bin/env python

"""
Split a given file into the specified number of files.

Order is preserved.

Author: Herman Kamper
Contact: h.kamper@sms.ed.ac.uk
Date: 2014
"""

from os import path
import argparse
import math
import os
import sys


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("input_fn", help="the input file")
    parser.add_argument("n_files", type=int, help="the number of files to split `input_fn` into")
    parser.add_argument(
        "output_dir", help="the split files are written here; "
        "a index (starting from 0) is added before the extension of the file, e.g. "
        "if `input_fn` is 'apples.list' then the first split file will be 'apples.0.list'"
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
    input_fn = args.input_fn
    n_files = args.n_files
    output_dir = args.output_dir

    if not path.isdir(output_dir):
        os.mkdir(output_dir)

    # Read the input file
    lines = open(input_fn).readlines()
    print "Number of lines in input:", len(lines)
    n_lines_per_split = int(math.ceil(float(len(lines))/n_files))
    print "Max number of lines per split:", n_lines_per_split

    basename, extension = path.splitext(path.split(input_fn)[-1])

    # Write lines to split files
    i_split = 1
    n_cur_lines = 0
    fn_cur = path.join(output_dir, basename + "." + str(i_split) + extension)
    f_cur = open(fn_cur, "w")
    for line in lines:
        f_cur.write(line)
        n_cur_lines += 1
        if n_cur_lines == n_lines_per_split:
            f_cur.close()
            print "Wrote " + str(n_cur_lines) + " lines to: " + fn_cur
            if i_split == n_files:
                break
            i_split += 1
            fn_cur = path.join(output_dir, basename + "." + str(i_split) + extension)
            f_cur = open(fn_cur, "w")
            n_cur_lines = 0
    f_cur.close()
    print "Wrote " + str(n_cur_lines) + " lines to: " + fn_cur


if __name__ == "__main__":
    main()
