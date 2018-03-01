import inspect


class Benchmark(object):
    """Base class for all benchmarks."""

    def __init__(self, *args, **kwargs):
        # Set pretty_name to function name, instead of the default
        # ``<module>.<class>.<function_name>``. This is a workaround
        # needed until ASV 0.3 release.
        members = inspect.getmembers(
            self.__class__,
            predicate=lambda x: inspect.ismethod(x) or inspect.isfunction(x))
        for (name, func) in members:
            func.pretty_name = name
