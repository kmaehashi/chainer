from functools import wraps
import inspect

import chainer
import cupy
import numpy

from .helper import _is_func
from .helper import parameterize
from .helper import synchronize


_backend_modes = [
    'gpu',
    'gpu-cudnn',
    'cpu',
    'cpu-ideep',
]


def backends(*modes):
    """Class decorator to parameterize the benchmark class with backends.

    This is a special form of :func:`parameterize` to parameterize the
    backend configuration. For all `time_*` functions and `setup` function
    in the class, this decorator:

    * wraps the function to be called with the Chainer configuration
      that corresponds to the current backend configuration.
    * wraps the function to perform CPU/GPU synchronization after the
      benchmark, when the current backend configuration uses GPU. The time
      taken for synchronization is counted as a elapsed time in the benchmark.
    * injects `xp` (`cupy` or `numpy` depending on the current configuration)
      as the first argument of the function so that benchmark code can use it
      to work with array modules with each backend.
    * provides access to `is_backend_*()` methods so that benchmark code can
      use it to change behavior depending on the backend configuration (e.g.,
      `if is_backend_gpu(): model.to_gpu()`).

    Note that `cpu-ideep` mode will automatically be removed if the current
    benchmark setup does not support it, e.g., when running benchmark
    against older Chainer version that does not support iDeep.
    """

    assert all([m in _backend_modes for m in modes])

    if not have_ideep() and 'cpu-ideep' in modes:
        modes.remove('cpu-ideep')

    def _wrap_class(klass):
        assert isinstance(klass, type)
        return _inject_backend_mode(klass, modes)

    return _wrap_class


def _inject_backend_mode(klass, modes):
    klass = parameterize([('mode', modes)], _head=True)(klass)
    members = inspect.getmembers(klass, predicate=_is_func)

    for (name, func) in members:
        if not (name == 'setup' or name.startswith('time_')):
            continue

        def _wrap_func(f):
            @wraps(f)
            def _wrapped_func(self, mode, *args, **kwargs):
                _benchmark_backend_gpu = False
                _benchmark_backend_ideep = False
                xp = numpy
                use_cudnn = 'never'
                use_ideep = 'never'

                target = f
                if mode.startswith('gpu'):
                    xp = cupy
                    _benchmark_backend_gpu = True
                    target = synchronize(target)
                    if 'cudnn' in mode:
                        use_cudnn = 'auto'
                elif 'ideep' in mode:
                    use_ideep = 'auto'
                    _benchmark_backend_ideep = True

                with _BackendConfig({
                        'use_cudnn': use_cudnn,
                        'use_ideep': use_ideep,
                        '_benchmark_backend_gpu': _benchmark_backend_gpu,
                        '_benchmark_backend_ideep': _benchmark_backend_ideep,
                        }):
                    target(self, xp, *args, **kwargs)

            return _wrapped_func
        setattr(klass, name, _wrap_func(func))

    return klass


class _BackendConfig(object):
    """Combines multiple Chainer configurations as one context manager."""

    def __init__(self, params):
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


def is_backend_gpu():
    """Returns True if the current backend is GPU."""

    return chainer.config._benchmark_backend_gpu


def is_backend_ideep():
    """Returns True if the current backend is iDeep."""

    return chainer.config._benchmark_backend_ideep


def have_ideep():
    """Tests if iDeep can be used in the current benchmark configuration.

    If you intend to write benchmark for iDeep outside of `backend` decorator,
    first make sure that iDeep is available using this function.
    This makes possible to run the same benchmark code over past versions of
    Chainer (prior to iDeep support).
    """

    try:
        import chainer.backends.intel64
    except ImportError:
        return False
    return chainer.backends.intel64.is_ideep_available()
