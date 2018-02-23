import chainer
from chainer import cuda
from chainer import optimizers

import cupy
import numpy

from .utils import parameterize, backends, is_backend_gpu, is_backend_ideep


@backends('gpu', 'gpu-cudnn', 'cpu', 'cpu-ideep')
class ConvnetOthers(object):
    def test_convolution_nd(self):
