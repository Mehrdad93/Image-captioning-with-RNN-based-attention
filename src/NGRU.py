from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class LayerRNNCell(tf.nn.rnn_cell.RNNCell):
  """Subclass of RNNCells that act like proper `tf.Layer` objects.

  For backwards compatibility purposes, most `RNNCell` instances allow their
  `call` methods to instantiate variables via `tf.get_variable`.  The underlying
  variable scope thus keeps track of any variables, and returning cached
  versions.  This is atypical of `tf.layer` objects, which separate this
  part of layer building into a `build` method that is only called once.

  Here we provide a subclass for `RNNCell` objects that act exactly as
  `Layer` objects do.  They must provide a `build` method and their
  `call` methods do not access Variables `tf.get_variable`.
  """

  def __call__(self, inputs, state, scope=None, *args, **kwargs):
    """Run this RNN cell on inputs, starting from the given state.

    Args:
      inputs: `2-D` tensor with shape `[batch_size, input_size]`.
      state: if `self.state_size` is an integer, this should be a `2-D Tensor`
        with shape `[batch_size, self.state_size]`.  Otherwise, if
        `self.state_size` is a tuple of integers, this should be a tuple
        with shapes `[batch_size, s] for s in self.state_size`.
      scope: optional cell scope.
      *args: Additional positional arguments.
      **kwargs: Additional keyword arguments.

    Returns:
      A pair containing:

      - Output: A `2-D` tensor with shape `[batch_size, self.output_size]`.
      - New state: Either a single `2-D` tensor, or a tuple of tensors matching
        the arity and shapes of `state`.
    """
    # Bypass RNNCell's variable capturing semantics for LayerRNNCell.
    # Instead, it is up to subclasses to provide a proper build
    # method.  See the class docstring for more details.
    return base_layer.Layer.__call__(self, inputs, state, scope=scope,
                                     *args, **kwargs)

@tf_export("NGRUCell")
class NGRUCell(LayerRNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).

  Note that this cell is not optimized for performance. Please use
  `tf.contrib.cudnn_rnn.CudnnGRU` for better performance on GPU, or
  `tf.contrib.rnn.GRUBlockCellV2` for better performance on CPU.

  Args:
    num_units: int, The number of units in the GRU cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
    projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such
      cases.
    dtype: Default dtype of the layer (default of `None` means use the type
      of the first input). Required when `build` is called before `call`.
    **kwargs: Dict, keyword named properties for common layer attributes, like
      `trainable` etc when constructing the cell from configs of get_config().
  """

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None,
               name=None,
               dtype=None,
               **kwargs):
    super(NGRUCell, self).__init__(
        _reuse=reuse, name=name, dtype=dtype, **kwargs)

    if context.executing_eagerly() and context.num_gpus() > 0:
      logging.warn("%s: Note that this cell is not optimized for performance. "
                   "Please use tf.contrib.cudnn_rnn.CudnnGRU for better "
                   "performance on GPU.", self)
    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    if activation:
      self._activation = activations.get(activation)
    else:
      self._activation = math_ops.tanh
    self._kernel_initializer = initializers.get(kernel_initializer)
    self._bias_initializer = initializers.get(bias_initializer)

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @tf_utils.shape_type_conversion
  def build(self, inputs_shape):
    if inputs_shape[-1] is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % str(inputs_shape))

    input_depth = inputs_shape[-1]
    self._gate_kernel = self.add_variable(
        "gates/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + self._num_units, 2 * self._num_units],
        initializer=self._kernel_initializer)
    self._gate_bias = self.add_variable(
        "gates/%s" % _BIAS_VARIABLE_NAME,
        shape=[2 * self._num_units],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else init_ops.constant_initializer(1.0, dtype=self.dtype)))
    self._candidate_kernel = self.add_variable(
        "candidate/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + self._num_units, self._num_units],
        initializer=self._kernel_initializer)
    self._candidate_bias = self.add_variable(
        "candidate/%s" % _BIAS_VARIABLE_NAME,
        shape=[self._num_units],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else init_ops.zeros_initializer(dtype=self.dtype)))

    self.built = True

  def call(self, inputs, state):
    """Gated recurrent unit (GRU) with nunits cells."""

    gate_inputs = math_ops.matmul(
        array_ops.concat([inputs, state], 1), self._gate_kernel)
    gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

    value = math_ops.sigmoid(gate_inputs)
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * state

    candidate = math_ops.matmul(
        array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
    candidate = nn_ops.bias_add(candidate, self._candidate_bias)

    c = self._activation(candidate)
    new_h = u * state + (1 - u) * c
    return new_h, new_h

  def get_config(self):
    config = {
        "num_units": self._num_units,
        "kernel_initializer": initializers.serialize(self._kernel_initializer),
        "bias_initializer": initializers.serialize(self._bias_initializer),
        "activation": activations.serialize(self._activation),
        "reuse": self._reuse,
    }
    base_config = super(NGRUCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
