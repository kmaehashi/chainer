#!/bin/bash -uex

COMMITS=$(git log --merges --first-parent --format=%H v4.0.0b4..master)

for CHAINER_COMMIT in ${COMMITS}; do
  CUPY_COMMIT=$(./find_cupy_version.py --commit ${CHAINER_COMMIT} --cupy-branch master)
  echo "** RUN: Chainer $CHAINER_COMMIT / CuPy $CUPY_COMMIT"
  ./run.sh ${CHAINER_COMMIT} ${CUPY_COMMIT} -vvv
done
