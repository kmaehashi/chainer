Chainer Benchmarks
==================

Benchmarking Chainer with Airspeed Velocity.

.. code-block:: py

    # Install Airspeed Velocity.
    pip install asv

    # Clone the CuPy source tree.
    # Then checkout the CuPy branch compatible with the Chainer branch you
    # are going to benchmark.
    git clone https://github.com/cupy/cupy.git
    cd cupy
    git checkout master
    cd ..

    # Remove the old benchmark environment.
    # This is only needed if you changed the CuPy branch to use.
    rm -rf env

    # Run benchmark for the specified branch (or commit).
    # The benchmark result will be stored under `results` directory.
    asv run --step 1 master

    # Compare the benchmarks results between two commits to see regression
    # and/or performance improvements in command line.
    alias git_commit='git show --format="%H"'
    asv compare $(git_commit v4.0.0b4) $(git_commit master)

    # Convert the results into HTML.
    # The result will be in `html` directory.
    asv publish

    # Start the HTTP server to browse HTML.
    asv preview
