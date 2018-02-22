_backend_modes = [
    'gpu',
    'gpu-cudnn',
    'cpu',
    'cpu-ideep',
]


def backends(modes):
    """Class decorator to parameterize the benchmark class with backends.

    This is a special form of :func:`parameterize` which injects an array
    backend and Chainer configurations to benchmark. Moreover, by using this
    decorator, CPU/GPU synchronization is automatically performed after
    running the GPU benchmark.

    This injects `xp` to the first argument of all `time_*` methods and
    `setup` method in the class.

    `cpu-ideep` mode will automatically be removed if the current benchmark
    configuration does not support it.
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
    members = inspect.getmembers(
        klass,
        predicate=lambda _: inspect.ismethod(_) or inspect.isfunction(_))

    for (name, func) in members:
        if not (name == 'setup' or name.startswith('time_')):
            continue
        def _wrap_func(f):
            @wraps(f)
            def wrapped_func(self, mode, *args, **kwargs):
                xp = cupy if mode.startswith('gpu') else numpy
                use_cudnn = 'auto' if 'cudnn' in mode else 'never'
                use_ideep = 'auto' if 'ideep' in mode else 'never'

                if xp is cupy:
                    f = synchronize(f)

                with _BackendConfig(use_cudnn=use_cudnn, use_ideep=use_ideep):
                    f(self, xp, *args, **kwargs)

            return wrapped_func
        setattr(klass, name, _wrap_func(func))

    return klass


class _BackendConfig(object):
    """Combines multiple Chainer configurations into one context.

    Based on :class:`chainer.testing.backend.BackendConfig`."""

    def __init__(self, **params):
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


def have_ideep():
    """Tests if iDeep can be used in the current benchmark configuration.

    If you intend to write benchmark for iDeep, first make sure that iDeep
    is available using this function. This makes possible to run the same
    benchmark code over past versions of Chainer (prior to iDeep support),
    which is important for detecting regresesion.
    """

    try:
        import chainer.backends.intel64
    except ImportError as e:
        return False
    return chainer.backends.intel64.is_ideep_available()
