import os

from asv.plugins.virtualenv import Virtualenv
from asv.console import log


class VolatileVirtualenv(Virtualenv):
    """Manages an volatile environment using virtualenv.

    This is basically the same as ASV's default "virtualenv" plugin, but uses
    environment variable (in addition to the benchmark configuration specified
    in the config file) to control when to invalidate benchmark environment
    cache.
    """

    tool_name = "volatile-virtualenv"

    def __init__(self, *args, **kwargs):
        key = os.environ.get('VOLATILE_VIRTUALENV_KEY', None)
        if key is None:
            log.info('No key specified for volatile virtualenv')
        else:
            log.info('Using volatile virtualenv with key: {}'.format(key))
        self._volatile_key = key

        super(VolatileVirtualenv, self).__init__(*args, **kwargs)

    @property
    def name(self):
        default_name = super(VolatileVirtualenv, self).name
        if self._volatile_key is None:
            return default_name
        return '{}-{}'.format(default_name, self._volatile_key)
