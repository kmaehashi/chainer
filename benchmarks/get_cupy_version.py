#!/usr/bin/env python

import re
import sys


# Take care of special cases.
CHAINER_CUPY_VERSION_MAP = {
    # Always use CuPy master branch for Chainer master branch.
    'master': 'master',

    # CuPy v2.1.0.1 (hotfix) instead of v2.1.0.
    'v3.1.0': 'v2.1.0.1',

    # CuPy v3 has been bumped to v4 starting from v4.0.0b1.
    'v4.0.0a1': 'v3.0.0a1',
}


def get_cupy_version_for(chainer):
    """Returns CuPy version required for the given Chainer version."""

    # Handle special cases.
    if chainer in CHAINER_CUPY_VERSION_MAP:
        return CHAINER_CUPY_VERSION_MAP[chainer]

    # Use standard rules for the rest.
    m = re.search('^v(\d)\.(.+)$', chainer)
    if m is None:
        raise ValueError(chainer)

    chainer_major = int(m.group(1))
    chainer_rest = m.group(2)
    if chainer_major <= 1:
        raise ValueError('Chainer v1 or earlier is unsupported')
    elif 2 <= chainer_major <= 3:
        # Chainer vN requires CuPy v(N-1).
        return 'v{}.{}'.format((chainer_major - 1), chainer_rest)
    else:
        # The same versioning as Chainer.
        return chainer


if __name__ == '__main__':
    print(get_cupy_version_for(sys.argv[1]))
