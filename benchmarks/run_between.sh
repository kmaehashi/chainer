#!/bin/bash -uex


COMMITS=$(git log --merges --first-parent --format=%H v4.0.0b3..v4.0.0b4)

for CHAINER_COMMIT in ${COMMITS}; do
  CUPY_COMMIT=$(./get_cupy_revision.sh ${CHAINER_COMMIT} master)
  echo "** RUN: Chainer $CHAINER_COMMIT / CuPy $CUPY_COMMIT"
  ./run.sh ${CHAINER_COMMIT} ${CUPY_COMMIT} -vvv
done
