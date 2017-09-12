all:
	python setup.py build_ext --inplace

clean:
	rm -rf build
	rm -rf speech_dtw/*.c
	rm -rf speech_dtw/*.so
	rm -rf speech_dtw/*.pyc

test:
	nosetests -v
