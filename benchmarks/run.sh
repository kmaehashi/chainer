#!/bin/bash -uex

CHAINER_COMMIT="${1}"; shift
CUPY_COMMIT="${1}"; shift

# Clone CuPy.
if [ ! -d cupy ]; then
  git clone https://github.com/cupy/cupy.git
fi

# Checkout the branch to use.
# Also run build to boost up installation.
pushd cupy
git checkout "${CUPY_COMMIT}"
popd

# Remove the old benchmark environment.
# This is needed to reinstall cloned CuPy branch to the environment.
rm -rf env

# Run the benchmark.
asv run --step 1 "$@" "${CHAINER_COMMIT}"
