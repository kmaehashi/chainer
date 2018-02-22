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

        klass.params = params
        klass.param_names = param_names

        return klass

    return _wrap_class
