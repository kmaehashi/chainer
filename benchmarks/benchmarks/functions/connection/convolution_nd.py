import functools
from operator import mul

import numpy

import chainer
import chainer.functions as F
from chainer.utils import conv

from ...utils import backends, config


class _ConvolutionNDBase(object):
    def setup(self):
        xp = self.xp

        # Test parameters
        self.dims = (256, 256, 256)
        self.cover_all = True
        self.c_contiguous = True
        self.x_dtype = xp.float32
        self.W_dtype = xp.float32

        # Prepare test data
        in_channels = 3
        out_channels = 2
        ndim = len(self.dims)
        ksize = (3,) * ndim
        self.stride = (2,) * ndim
        self.pad = (1,) * ndim

        W_scale = numpy.sqrt(1. / functools.reduce(mul, ksize, in_channels))
        W_shape = (out_channels, in_channels) + ksize
        self.W = numpy.random.normal(0, W_scale, W_shape).astype(self.W_dtype)
        self.b = numpy.random.uniform(-1, 1, out_channels).astype(self.x_dtype)

        x_shape = (2, 3) + self.dims
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(self.x_dtype)
        gy_shape = (2, 2) + tuple(
            conv.get_conv_outsize(d, k, s, p, cover_all=self.cover_all)
            for (d, k, s, p) in zip(self.dims, ksize, self.stride, self.pad))
        self.gy = numpy.random.uniform(-1, 1, gy_shape).astype(self.x_dtype)

        self.ggx = numpy.random.uniform(-1, 1, self.x.shape).astype(
            self.x_dtype)
        self.ggW = numpy.random.uniform(-1, 1, self.W.shape).astype(
            self.W_dtype)
        self.ggb = numpy.random.uniform(-1, 1, self.b.shape).astype(
            self.x_dtype)

        # Transfer test data to device.
        self.x = chainer.Variable(xp.asarray(self.x))
        self.W = chainer.Variable(xp.asarray(self.W))
        self.b = chainer.Variable(xp.asarray(self.b))

    def time_forward(self):
        F.convolution_nd(
            self.x, self.W, self.b, stride=self.stride,
            pad=self.pad, cover_all=self.cover_all)


@backends('gpu', 'cpu', 'cpu-ideep')
@config('autotune', [False])
class ConvolutionND(_ConvolutionNDBase):
    pass


@backends('gpu-cudnn')
@config('autotune', [True, False])
class ConvolutionNDCuDNN(_ConvolutionNDBase):
    pass
