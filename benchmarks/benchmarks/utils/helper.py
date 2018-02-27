from functools import wraps
import inspect

import cupy


def _is_func(target):
    return inspect.ismethod(target) or inspect.isfunction(target)


def synchronize(target):
    """Decorator to perform CPU/GPU synchronization.

    This decorator can be applied to both classes and functions.
    """

    if isinstance(target, type):
        klass = target
        members = inspect.getmembers(klass, predicate=_is_func)
        for (name, func) in members:
            if not (name == 'setup' or name.startswith('time_')):
                continue
            setattr(klass, name, _synchronize_func(func))
        return klass
    elif _is_func(target):
        return _synchronize_func(target)
    else:
        raise TypeError('cannot apply decorator to {}'.format(target))


def _synchronize_func(func):
    @wraps(func)
    def _wrapped_func(*args, **kwargs):
        event = cupy.cuda.stream.Event()
        func(*args, **kwargs)
        print('Synchronization start...')
        event.synchronize()
        print('Synchronization done.')
    return _wrapped_func


def parameterize(args, _head=False):
    """Class decorator to parameterize the ASV benchmark class.

    Due to the limitation of ASV, parameters cannot be sparse.
    See the example below for the usage.

    >>> @parameterize(
    ...     ('batchsize', [32, 64, 128]),
    ...     ('n_gpus', [1, 2]),
    ... )
    ... class MyBenchmark(object):
    ...     def time_all(self, batchsize, n_gpus):
    ...         ...

    """

    def _wrap_class(klass):
        assert isinstance(klass, type)

        params = [arg[1] for arg in args]
        param_names = [arg[0] for arg in args]

        orig_params = getattr(klass, 'params', [])
        orig_param_names = getattr(klass, 'param_names', [])

        if 0 < len(orig_params):
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

        setattr(klass, 'params', params)
        setattr(klass, 'param_names', param_names)

        return klass

    return _wrap_class
