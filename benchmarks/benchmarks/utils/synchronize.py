from functools import wraps
import inspect

import cupy


def synchronize(target):
    """Decorator to perform CPU/GPU synchronization.

    This decorator can be applied to both classes and functions.
    """

    if isinstance(target, type):
        klass = target
        members = inspect.getmembers(
            klass,
            predicate=lambda _: (
                inspect.ismethod(_) or inspect.isfunction(_)))
        for (name, func) in members:
            if not (name == 'setup' or name.startswith('time_')):
                continue
            setattr(klass, name, _synchronize_func(func))
        return klass
    else:
        return _synchronize_func(target)


def _synchronize_func(func):
    assert inspect.ismethod(func) or inspect.isfunction(func)))

    @wraps(func)
    def wrapped_func(*args, **kwargs):
        event = cupy.cuda.stream.Event()
        func(*args, **kwargs)
        event.synchronize()
    return wrapped_func
