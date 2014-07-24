"""
Author: Herman Kamper
Contact: h.kamper@sms.ed.ac.uk
Date: 2014
"""

import numpy as np
import numpy.testing as npt

from speech_dtw import _dtw


def test_dtw():
    s2 = [
        71, 73, 75, 80, 80, 80, 78, 76, 75, 73, 71, 71, 71, 73, 75, 76, 76, 68]
    t2 = [69, 69, 73, 75, 79, 80, 79, 78, 76, 73, 72, 71, 70, 70, 69]

    path_expected = [
        (0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
        (8, 8), (9, 9), (10, 10), (11, 11), (12, 11), (13, 11), (14, 11), (15,
        12), (16, 13), (17, 14)]
    cost_expected = 27.0

    path, cost_mat = _dtw.dtw(s2, t2)
    print path

    assert path == path_expected
    assert abs(cost_mat[-1, -1] - cost_expected) < EPSILON

