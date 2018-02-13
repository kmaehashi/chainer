import chainer

import cupy

from ..utils import xp, parameterize


class _ConvnetBase(object):
    def setup(self, xp, arch, batchsize):
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

        self.model = model

        workspace_size = int(1 * 2**30)
        chainer.cuda.set_max_workspace_size(workspace_size)

    def time_all(self, xp, arch, batchsize):
        model = self.model

        optimizer = chainer.optimizers.SGD(lr=0.01)
        optimizer.setup(model)

        chainer.config.train = True
        chainer.config.use_cudnn = 'always'

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


@xp(numpy=False)
@parameterize([
    ('arch', ['vgga']),
    ('batchsize', [64]),
])
class ConvnetVGGA(_ConvnetBase):
    pass


@xp(numpy=False)
@parameterize([
    ('arch', ['alexnet', 'googlenet', 'overfeat']),
    ('batchsize', [128]),
])
class ConvnetOthers(_ConvnetBase):
    pass
