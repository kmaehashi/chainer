#!/bin/bash -uex

function run_benchmark() {
  CHAINER_COMMIT="${1}"; shift
  CUPY_COMMIT="${1}"; shift

  # Clone CuPy.
  if [ ! -d cupy ]; then
    git clone https://github.com/cupy/cupy.git
  fi

  # Checkout the branch to use.
  pushd cupy
  git checkout "${CUPY_COMMIT}"
  popd

  # Run the benchmark.
  # The benchmark environment depends on ${CUPY_COMMIT}.
  export VOLATILE_VIRTUALENV_KEY="${CUPY_COMMIT}"
  asv run --step 1 "$@" "${CHAINER_COMMIT}"
}

run_benchmark "$@"
