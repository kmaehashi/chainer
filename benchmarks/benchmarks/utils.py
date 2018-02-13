from functools import wraps
import inspect

import numpy
import cupy


def parameterize(args, _head=False):
    """Decorator to parameterize the benchmark class."""

    def f(klass):
        params = [arg[1] for arg in args]
        param_names = [arg[0] for arg in args]

        orig_params = getattr(klass, 'params', [])
        orig_param_names = getattr(klass, 'param_names', [])

        if orig_params:
            if not isinstance(orig_params[0], (tuple, list)):
                orig_params = [orig_params]
                if len(orig_param_names) == 0:
                    orig_param_names = ['param']
                assert len(orig_param_names) == 1
        else:
            assert len(orig_param_names) == 0

        if _head:
            params += orig_params
            param_names += orig_param_names
        else:
            params = orig_params + params
            param_names = orig_param_names + param_names

        assert len(params) == len(param_names)

        klass.params = params
        klass.param_names = param_names

        return klass

    return f


def xp(cupy=True, numpy=True):
    """Decorator to parameterize the benchmark class with xp.

    This is a special case of :func:`parametersize` decorator to inject
    CuPy/NumPy module as a benchmark parameter. This is needed because of the
    following reasons:

    * CPU/GPU synchronization must be performed after CuPy benchmark.
    * Directly parameterizing modules makes HTML view dirty.
    """

    modules = []
    if cupy:
        modules.append('cupy')
    if numpy:
        modules.append('numpy')

    def _wrap_class(klass):
        assert isinstance(klass, type)
        return _inject_xp(klass, modules)

    return _wrap_class


def _inject_xp(klass, modules):
    klass = parameterize([('xp', modules)], _head=True)(klass)

    # Wrap functions to overwrite parameters given in string ('cupy', 'numpy')
    # with the actual module reference.
    members = inspect.getmembers(
        klass,
        predicate=lambda _: inspect.ismethod(_) or inspect.isfunction(_))

    for (name, method) in members:
        if name == 'setup':
            def _wrap_method(target):
                @wraps(target)
                def wrapped_setup(self, xp, *args, **kwargs):
                    if xp == 'cupy':
                        xp = cupy
                        event = cupy.cuda.stream.Event()
                    elif xp == 'numpy':
                        xp = numpy
                        event = None
                    else:
                        raise AssertionError()

                    target(self, xp, *args, **kwargs)

                    assert not hasattr(self, '_xp')
                    assert not hasattr(self, '_cupy_event')
                    self._xp = xp
                    self._cupy_event = event
                return wrapped_setup

        elif name.startswith('time_'):
            def _wrap_method(target):
                @wraps(target)
                def wrapped_time_func(self, xp, *args, **kwargs):
                    xp = self._xp
                    event = self._cupy_event
                    del self._xp
                    del self._cupy_event

                    target(self, xp, *args, **kwargs)

                    if event:
                        event.synchronize()
                return wrapped_time_func

        else:
            continue

        setattr(klass, name, _wrap_method(method))

    return klass
