"""
Kaldi utility functions.

Author: Herman Kamper
Contact: h.kamper@sms.ed.ac.uk
Date: 2014
"""

import numpy as np
import re


def read_kaldi_ark(ark_fn):
    """Read a Kaldi archive (in text format) and return it in a dict."""
    ark_dict = {}
    lines = open(ark_fn).readlines()
    for line in lines:
        line = line.strip(" \n")
        if line[-1] == "[":
            cur_id = line.split()[0]
            cur_mat = []
        elif "]" in line:
            line = line.strip("]")
            cur_mat.append([float(i) for i in line.split()])
            ark_dict[cur_id] = np.array(cur_mat)
        else:
            cur_mat.append([float(i) for i in line.split()])
    return ark_dict


def write_kaldi_ark(ark_dict, ark_fn):
    """
    Write the Kaldi archive dict `ark_dict` to `ark_fn` in text format.
    """
    np.set_printoptions(linewidth=np.nan, threshold=np.nan)
    f = open(ark_fn, "w")
    for cur_id in ark_dict:
        cur_mat = ark_dict[cur_id]
        f.write(cur_id + "  [\n")
        f.write(re.sub("(\]\])|(\])|(\ \[)|(\[\[)", " ", str(cur_mat)))
        f.write("]\n")
    f.close()
    np.set_printoptions(linewidth=75, threshold=1000)
