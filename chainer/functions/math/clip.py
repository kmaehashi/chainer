import numpy

from chainer import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


class Clip(function_node.FunctionNode):
    """Clips (limits) elements of input variable."""

    def __init__(self, x_min, x_max):
        if not isinstance(x_min, float):
            raise TypeError('x_min must be float value')
        if not isinstance(x_max, float):
            raise TypeError('x_max must be float value')
        # x_min must be lesser than x_max.
        assert x_min < x_max
        self.x_min = x_min
        self.x_max = x_max

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types
        type_check.expect(x_type.dtype.kind == 'f')

    def forward_cpu(self, x):
        self.retain_inputs((0,))
        return utils.force_array(
            numpy.clip(x[0], self.x_min, self.x_max)),

    def forward_gpu(self, x):
        self.retain_inputs((0,))
        return cuda.cupy.clip(x[0], self.x_min, self.x_max),

    def backward(self, indexes, grad_outputs):
        x = self.get_retained_inputs()
        return ClipGrad(self.x_min, self.x_max).apply((x[0], grad_outputs[0]))


class ClipGrad(function_node.FunctionNode):

    def __init__(self, x_min, x_max):
        if not isinstance(x_min, float):
            raise TypeError('x_min must be float value')
        if not isinstance(x_max, float):
            raise TypeError('x_max must be float value')
        # x_min must be lesser than x_max.
        assert x_min < x_max
        self.x_min = x_min
        self.x_max = x_max

    def forward_cpu(self, inputs):
        self.retain_outputs((0,))
        x, gy = inputs
        return utils.force_array(
            gy * (self.x_min < x) * (x < self.x_max)),

    def forward_gpu(self, inputs):
        self.retain_outputs((0,))
        x, gy = inputs
        gx = cuda.elementwise(
            'T x, T gy, T x_min, T x_max', 'T gx',
            'gx = ((x > x_min) & (x < x_max)) ? gy : T(0)',
            'clip_bwd')(x, gy, self.x_min, self.x_max)
        return gx,

    def backward(self, indexes, grad_outputs):
        gx = self.get_retained_outputs()[0]
        return 0, grad_outputs[0] if gx > 0 else 0


def clip(x, x_min, x_max):
    """Clips (limits) elements of input variable.

    Given an interval ``[x_min, xmax]``, elements outside the interval are
    clipped to the interval edges.

    Args:
        x (~chainer.Variable): Input variable to be clipped.
        x_min (float): Minimum value.
        x_max (float): Maximum value.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Clip(x_min, x_max).apply((x,))[0]
