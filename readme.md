Dynamic Time Warping for Speech Processing
==========================================

Overview
--------
Tools (functions and scripts) for performing and evaluating speech processing
tasks using dynamic time warping (DTW).


Dependencies
------------
- [Cython](http://cython.org/)


Building and testing the code
-----------------------------
- Build the Cython components: `make`
- Run unit tests: `make test`
- Clean the code: `make clean`


Examples
--------
Examples are given in the `examples/` directory. The script
`examples/rnd_example.py` illustrates standard DTW and compares a pure Python
implementation to the Cython implementation.

The IPython notebook `examples/speech_dtw_qbe.ipynb` illustrates how to do
(very basic) query-by-example (QbE) search using standard DTW.
