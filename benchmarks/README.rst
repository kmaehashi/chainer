Chainer Benchmarks
==================

Benchmarking Chainer with Airspeed Velocity.

Requirements
------------

* ``asv``
* ``Cython`` (to build CuPy)

Usage
-----

.. code-block:: sh

    # Run benchmark against target commit-ish of Chainer and CuPy.
    # Note that specified versions must be a compatible combination.
    # You can use `get_cupy_version.py` helper tool to get appropriate CuPy
    # version for the given Chainer version.
    ./run.sh master master
    ./run.sh v4.0.0b4 v4.0.0b4

    # Compare the benchmark results between two commits to see regression
    # and/or performance improvements in command line.
    alias git_commit='git show --format="%H"'
    asv compare $(git_commit v4.0.0b4) $(git_commit master)

    # Convert the results into HTML.
    # The result will be in `html` directory.
    asv publish

    # Start the HTTP server to browse HTML.
    asv preview
