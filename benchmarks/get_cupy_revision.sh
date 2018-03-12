#!/bin/bash -uex

CHAINER_COMMIT="${1}"
CUPY_BRANCH="${2}"

cd chainer
CHAINER_COMMIT_TIME="$(git show --format="%ct" "${CHAINER_COMMIT}")"
cd ..

cd cupy
CUPY_COMMIT="$(git log --merges --first-parent --max-count 1 --until "${CHAINER_COMMIT_TIME}" --format="%H" "${CUPY_BRANCH}")"
cd ..

echo "${CUPY_COMMIT}"
