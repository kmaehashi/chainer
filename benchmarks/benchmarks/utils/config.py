from functools import wraps
import inspect

import chainer

from .helper import _is_func
from .helper import parameterize


def config(key, values):
    """Class decorator to parameterize the Chainer configuration.

    This is a special form of :func:`parameterize` to parameterize the
    Chainer configuration. For all `time_*` functions and `setup` function
    in the class, this decorator:

    * wraps the function to be called with the specified Chainer
      configuration set.

    This decorator adds parameter axis with the name of the key.

    >>> @config('autotune', [True, False])
    ... class ConvolutionBenchmark(object):
    ...     def time_benchmark(self):
    ...         ...
    """

    def _wrap_class(klass):
        assert isinstance(klass, type)
        return _inject_config(klass, key, values)

    return _wrap_class


def _inject_config(klass, key, values):
    klass = parameterize([(key, values)], _head=True)(klass)
    members = inspect.getmembers(klass, predicate=_is_func)

    for (name, func) in members:
        if not (name == 'setup' or name.startswith('time_')):
            continue

        def _wrap_func(f):
            @wraps(f)
            def _wrapped_func(self, _config_value, *args, **kwargs):
                with chainer.using_config(key, _config_value):
                    f(self, *args, **kwargs)

            return _wrapped_func
        setattr(klass, name, _wrap_func(func))

    return klass
