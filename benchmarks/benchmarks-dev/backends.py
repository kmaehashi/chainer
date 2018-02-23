import chainer
import cupy
import numpy


class BackendConfig(object):

    def __init__(**params):
        self._params = params
        self._contexts = []

    def __enter__(self):
        self._contexts = [
            chainer.using_config(k, v) for (k, v) in self._params.items()
        ]
        for c in self._contexts:
            c.__enter__()
        return self

    def __exit__(self, typ, value, traceback):
        for c in reversed(self._contexts):
            c.__exit__(typ, value, traceback)
