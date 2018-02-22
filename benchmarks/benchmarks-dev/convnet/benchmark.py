import chainer
from chainer import cuda
from chainer import optimizers

import cupy
import numpy

import ideep4py
from ..utils import parameterize, have_ideep


class _ConvnetBase(object):
    timeout = 600
    number = 1

    def setup(self, arch, batchsize, mode):
        xp = cupy if mode == 'gpu' else numpy
        ideep = True if mode == 'cpu-ideep' else False

        if arch == 'alexnet':
            from .nets import alex
            model = alex.Alex()
        elif arch == 'googlenet':
            from .nets import googlenet
            model = googlenet.GoogLeNet()
        elif arch == 'vgga':
            from .nets import vgga
            model = vgga.vgga()
        elif arch == 'overfeat':
            from .nets import overfeat
            model = overfeat.overfeat()
        else:
            raise ValueError('Invalid architecture name')

        if xp is cupy:
            model.to_gpu()
        elif ideep:
            assert have_ideep()
            model.to_intel64()

        # Setup optimizer
        optimizer = optimizers.SGD(lr=0.01)
        optimizer.setup(model)

        workspace_size = int(1 * 2**30)
        chainer.cuda.set_max_workspace_size(workspace_size)

        chainer.config.train = True
        chainer.config.use_cudnn = 'always'

        # Trainer
        data = xp.ndarray((batchsize, 3, model.insize,
                           model.insize), dtype=xp.float32)
        data.fill(33333)

        x = xp.asarray(data)

        if arch == 'googlenet':
            out1, out2, out3 = model.forward(x)
            out = out1 + out2 + out3
        else:
            out = model.forward(x)

        out.zerograd()
        out.grad.fill(3)
        model.cleargrads()
        out.backward()

        self._x = x
        self._model = model
        self._out = out
        self._use_ideep = 'auto' if ideep else 'never'

    def time_forward(self, arch, batchsize, mode):
        with chainer.using_config('use_ideep', self._use_ideep):
            self._model.forward(self._x)

    def time_backward(self, arch, batchsize, mode):
        with chainer.using_config('use_ideep', self._use_ideep):
            self._out.backward()


_modes = ['gpu', 'cpu'] + (['cpu-ideep'] if have_ideep() else [])

@parameterize([
    ('arch', ['vgga']),
    ('batchsize', [64]),
    ('mode', _modes),
])
class ConvnetVGGA(_ConvnetBase):
    pass


@parameterize([
    ('arch', ['alexnet', 'googlenet', 'overfeat']),
    ('batchsize', [128]),
    ('mode', _modes),
])
class ConvnetOthers(_ConvnetBase):
    pass
