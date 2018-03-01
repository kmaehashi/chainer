from utils import *

from . import Benchmark

@backends('gpu-cudnn')
@config('baz', ['baz-x','baz-y'])
@config('bar', ['bar-1'])
@parameterize([
    ('batchsize', ['BATCH_SIZE']),
    ('n_gpus', ['N_GPU']),
])
class MyBenchmark(object):
    def time_it(self, batchsize, n_gpus):
        print('final param names', self.param_names)
        print('final params', self.params)
        print(self.xp)
        print(batchsize)
        print(n_gpus)
        import chainer
        chainer.config.show()
