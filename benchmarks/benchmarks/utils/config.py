from functools import wraps
import inspect

import chainer

from .helper import _is_func
from .helper import parameterize


def config(key, values, axis=None):
    """Class decorator to parameterize the Chainer configuration.

    This is a specialized form of `parameterize` decorator to parameterize
    the Chainer configuration. For all `time_*` functions and `setup` function
    in the class, this decorator wraps the function to be called inside the
    context where specified Chainer configuration set.

    Note that the configuration will not be parameterized if Chainer used in
    the benchmark does not support it.

    This decorator adds parameter axis with the name of the configuration
    by default. You can change the axis name by passing axis parameter.

    Example of usage is as follows:

    >>> @config('autotune', [True, False])
    ... class ConvolutionBenchmark(object):
    ...     def time_benchmark(self):
    ...         ...
    """

    axis = key if axis is None else axis

    def _wrap_class(klass):
        assert isinstance(klass, type)
        if not hasattr(chainer.config, key):
            print(
                '''Configuration '{}' unknown to this version of '''
                '''Chainer; will not be parameterized'''.format(key))
            return klass
        return _inject_config(klass, axis, key, values)

    return _wrap_class


def _inject_config(klass, axis, key, values):
    klass = parameterize([(axis, values)])(klass)
    members = inspect.getmembers(klass, predicate=_is_func)

    for (name, func) in members:
        if not (name == 'setup' or name.startswith('time_')):
            continue

        def _wrap_func(f):
            @wraps(f)
            def _wrapped_func(self, value, *args, **kwargs):
                with chainer.using_config(key, value):
                    f(self, *args, **kwargs)

            return _wrapped_func
        setattr(klass, name, _wrap_func(func))

    return klass
