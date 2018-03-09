#!/bin/bash -uex

function git_commit() {
  git show --format="%H" "$@"
}

function run_benchmark() {
  CHAINER_COMMIT="${1}"; shift
  CUPY_COMMIT="${1}"; shift

  # Ensure to use commit hash in case CHAINER_COMMIT is a branch.
  CHAINER_COMMIT="$(git_commit ${CHAINER_COMMIT})"

  # Clone CuPy.
  if [ ! -d cupy ]; then
    git clone https://github.com/cupy/cupy.git
  fi

  # Checkout the branch to use.
  pushd cupy
  git checkout "${CUPY_COMMIT}"

  # Ensure to use commit hash in case CUPY_COMMIT is a branch.
  CUPY_COMMIT="$(git_commit ${CUPY_COMMIT})"
  popd

  # Run the benchmark.
  # The benchmark environment depends on ${CUPY_COMMIT}.
  export VOLATILE_VIRTUALENV_KEY="${CUPY_COMMIT}"
  export PIP_NO_DEPS=True
  export PIP_VERBOSE=True
  export PIP_LOG=pip.log
  asv run --step 1 "$@" "${CHAINER_COMMIT}"
}

run_benchmark "$@"
