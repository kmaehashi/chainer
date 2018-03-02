from functools import wraps
import inspect
import warnings

import chainer

from benchmarks.utils.helper import _is_func
from benchmarks.utils.helper import parameterize


def config(key, values, axis=None):
    """Class decorator to parameterize the Chainer configuration.

    This is a specialized form of `parameterize` decorator to parameterize
    the Chainer configuration. For all `time_*` functions and `setup` function
    in the class, this decorator wraps the function to be called inside the
    context where specified Chainer configuration set.

    This decorator adds parameter axis with the name of the configuration
    by default. You can change the axis name by passing axis parameter.

    Note that the configuration will automatically be removed if Chainer used
    in the current benchmark setup does not support it.

    You cannot apply `parameterize` decorator to the class already decorated
    by this decorator.  If you want to use `parameterize` along with this
    decorator, make `parameterize` the most inner (i.e., the closest to the
    class declaration) decorator.

    Example of usage is as follows:

    >>> @config('autotune', [True, False])
    ... class ConvolutionBenchmark(object):
    ...     def time_benchmark(self):
    ...         ...
    """

    axis = key if axis is None else axis

    def _wrap_class(klass):
        assert isinstance(klass, type)
        return _inject_config(klass, axis, key, values)

    return _wrap_class


def _inject_config(klass, axis, key, values):
    klass = parameterize([(axis, values)])(klass)
    members = inspect.getmembers(klass, predicate=_is_func)

    # `setup` method is mandatory to inject backends to skip axis.
    if not hasattr(klass, 'setup'):
        def _setup(self, *args, **kwargs):
            pass
        klass.setup = _setup

    for (name, func) in members:
        if not (name == 'setup' or name.startswith('time_')):
            continue

        def _wrap_func(f):
            @wraps(f)
            def _wrapped_func(self, value, *args, **kwargs):
                if not hasattr(chainer.config, key):
                    # Raise in `setup` to skip this parameter axis.
                    warnings.warn(
                        'Configuration key "{}" unknown to this version of '
                        'Chainer'.format(key))
                    raise NotImplementedError

                with chainer.using_config(key, value):
                    f(self, *args, **kwargs)

            return _wrapped_func
        setattr(klass, name, _wrap_func(func))

    return klass
