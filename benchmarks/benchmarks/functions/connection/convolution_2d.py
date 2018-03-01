import numpy

import chainer
import chainer.functions as F

from ...utils import backends, config


class _Convolution2DBase(object):
    def setup(self):
        xp = self.xp

        # Test parameters
        self.c_contiguous = True
        self.cover_all = True
        self.x_dtype = xp.float32
        self.dilate = 1
        self.group = 1

        # Prepare test data
        batches = 2
        in_channels_a_group = 3
        out_channels_a_group = 2
        in_channels = in_channels_a_group * self.group
        out_channels = out_channels_a_group * self.group
        kh, kw = (3, 3)
        self.stride = 2
        self.pad = (int(kh / 2) * self.dilate, int(kw / 2) * self.dilate)
        self.use_cudnn = 'always'
        self.W = numpy.random.normal(
            0, numpy.sqrt(1. / (kh * kw * in_channels_a_group)),
            (out_channels, in_channels_a_group, kh, kw)).astype(self.W_dtype)
        self.b = numpy.random.uniform(
            -1, 1, out_channels).astype(self.x_dtype)

        self.x = numpy.random.uniform(
            -1, 1, (batches, in_channels, 4, 3)).astype(self.x_dtype)

        # Transfer data to device.
        self.x = chainer.Variable(xp.asarray(self.x))
        self.W = chainer.Variable(xp.asarray(self.W))
        self.b = chainer.Variable(xp.asarray(self.b))

    def time_forward(self):
       F.convolution_2d(
            self.x, self.W, self.b, stride=self.stride, pad=self.pad,
            cover_all=self.cover_all, dilate=self.dilate, group=self.group)


@backends('gpu', 'cpu', 'cpu-ideep')
@config('autotune', [False])
class NoCuDNN(_Convolution2DBase):
    pass


@backends('gpu-cudnn')
@config('autotune', [True, False])
class CuDNN(_Convolution2DBase):
    pass
