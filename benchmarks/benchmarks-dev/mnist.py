#!/usr/bin/env python

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

from .utils import parameterize, have_ideep


class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units)
            self.l2 = L.Linear(None, n_units)
            self.l3 = L.Linear(None, n_out)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


class MLPApplication(object):

    def main(self, units=1000, epoch=20, batchsize=100, gpu=-1, ideep=False):
        model = L.Classifier(MLP(units, 10))

        if 0 <= gpu:
            assert not ideep
            chainer.cuda.get_device_from_id(gpu).use()
            model.to_gpu()
        elif ideep:
            assert have_ideep()
            model.to_intel64()

        optimizer = chainer.optimizers.MomentumSGD().setup(model)

        train, test = chainer.datasets.get_mnist()
        train_iter = chainer.iterators.SerialIterator(
            train, batchsize)
        test_iter = chainer.iterators.SerialIterator(
            test, batchsize, repeat=False, shuffle=False)

        updater = training.updaters.StandardUpdater(
            train_iter, optimizer, device=gpu)
        trainer = training.Trainer(updater, (epoch, 'epoch'))
        trainer.extend(extensions.Evaluator(test_iter, model, device=gpu))
        trainer.extend(extensions.LogReport())

        with chainer.using_config('use_ideep', 'auto' if ideep else 'never'):
            trainer.run()


_modes = ['gpu', 'cpu'] + (['cpu-ideep'] if have_ideep() else [])

@parameterize([
    ('mode', _modes),
    ('units', [10, 100, 200, 300, 500]),
])
class TimeMLP(object):
    timeout = 360

    def time_mlp(self, mode, units):
        app = MLPApplication()
        gpu = 0 if mode == 'gpu' else -1
        ideep = True if mode == 'cpu-ideep' else False
        app.main(units=units, epoch=1, gpu=gpu, ideep=ideep)
