import chainer
from chainer import cuda
from chainer import optimizers

import functools

from .utils import parameterize, backends


class _ConvolutionNDBase(object):
    def setup(self, xp):
        # From test parameters:
        self.dims = (3, 4, 3)
        self.cover_all = True
        self.c_contiguous = True
        self.x_dtype = xp.float32
        self.W_dtype = xp.float32

        # From test setup:
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

    def time_convolution_nd(self, xp):
        F.convolution_nd(
            self.x, W_gpu, b_gpu, stride=self.stride,
            pad=self.pad,
            cover_all=self.cover_all)


@backends('gpu', 'cpu', 'cpu-ideep')
@config('autotune', [False])
class ConvolutionND(_ConvolutionNDBase):
    pass


@backends('gpu-cudnn')
@config('autotune', [True, False])
class ConvolutionNDCuDNN(_ConvolutionNDBase):
    pass
