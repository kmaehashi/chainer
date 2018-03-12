#!/bin/bash -uex

TAGS="
master
v4.0.0b4
v4.0.0b3
v4.0.0b2
v4.0.0b1
v3.4.0
v3.3.0
v3.2.0
v3.1.0
"

for CHAINER_TAG in ${TAGS}; do
  CUPY_TAG=$(./get_cupy_version.py ${CHAINER_TAG})
  ./run.sh ${CHAINER_TAG} ${CUPY_TAG} -vvv
done
