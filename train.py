# Copyright 2021 Calico LLC
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import os
import glob
import json
import functools
import inspect
from pathlib import Path

import tensorflow as tf
import sonnet as snt
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, Optional, Text, Union, Iterable, List

import wandb

# attribute

# Enformer tensorflow code was directly taken and modified for distributed training
# https://github.com/deepmind/deepmind-research/tree/master/enformer

# Genetic augmentation code was taken from
# https://github.com/calico/basenji/blob/84c681a4b02f592a3de90799cee7f17d96f81ef8/basenji/archive/augmentation.py

# constants

NUM_CORES_ENFORCE = 64  # using v3-64

SEQUENCE_LENGTH = 196_608
BIN_SIZE = 128
TARGET_LENGTH = 896

# assert TPUs

tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu = 'enformer')
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_strategy = snt.distribute.TpuReplicator(tpu)

num_cores =  tpu_strategy.num_replicas_in_sync
assert num_cores == NUM_CORES_ENFORCE, f'must betraining on {num_cores} cores'

# classes

class MultiheadAttention(snt.Module):
  """Multi-head attention."""

  def __init__(self,
               value_size: int,
               key_size: int,
               num_heads: int,
               scaling: bool = True,
               attention_dropout_rate: float = 0.1,
               relative_positions: bool = False,
               relative_position_symmetric: bool = False,
               relative_position_functions: Optional[List[str]] = None,
               num_relative_position_features: Optional[int] = None,
               positional_dropout_rate: float = 0.1,
               zero_initialize: bool = True,
               initializer: Optional[snt.initializers.Initializer] = None,
               name: str = None):
    """Creates a MultiheadAttention module.

    Args:.prefetch(2)
      value_size: The size of each value embedding per head.
      key_size: The size of each key and query embedding per head.
      num_heads: The number of independent queries per timestep.
      scaling: Whether to scale the attention logits.
      attention_dropout_rate: Dropout rate for attention logits.
      relative_positions: Whether to use TransformerXL style relative attention.
      relative_position_symmetric: If True, the symmetric version of basis
        functions will be used. If False, a symmetric and asymmetric versions
        will be use.
      relative_position_functions: List of function names used for relative
        positional biases.
      num_relative_position_features: Number of relative positional features
        to compute. If None, `value_size * num_heads` is used.
      positional_dropout_rate: Dropout rate for the positional encodings if
        relative positions are used.
      zero_initialize: if True, the final linear layer will be 0 initialized.
      initializer: Initializer for the projection layers. If unspecified,
        VarianceScaling is used with scale = 2.0.
      name: Name of module.
    """
    super().__init__(name=name)
    self._value_size = value_size
    self._key_size = key_size
    self._num_heads = num_heads
    self._attention_dropout_rate = attention_dropout_rate
    self._scaling = scaling
    self._relative_positions = relative_positions
    self._relative_position_symmetric = relative_position_symmetric
    self._relative_position_functions = relative_position_functions
    if num_relative_position_features is None:
      # num_relative_position_features needs to be divisible by the number of
      # relative positional functions *2 (for symmetric & asymmetric version).
      divisible_by = 2 * len(self._relative_position_functions)
      self._num_relative_position_features = (
          (self._value_size // divisible_by) * divisible_by)
    else:
      self._num_relative_position_features = num_relative_position_features
    self._positional_dropout_rate = positional_dropout_rate

    self._initializer = initializer
    if self._initializer is None:
      self._initializer = snt.initializers.VarianceScaling(scale=2.0)

    key_proj_size = self._key_size * self._num_heads
    embedding_size = self._value_size * self._num_heads

    self._q_layer = snt.Linear(
        key_proj_size,
        name='q_layer',
        with_bias=False,
        w_init=self._initializer)
    self._k_layer = snt.Linear(
        key_proj_size,
        name='k_layer',
        with_bias=False,
        w_init=self._initializer)
    self._v_layer = snt.Linear(
        embedding_size,
        name='v_layer',
        with_bias=False,
        w_init=self._initializer)
    w_init = snt.initializers.Constant(1e-8) if zero_initialize else self._initializer
    self._embedding_layer = snt.Linear(
        embedding_size,
        name='embedding_layer',
        w_init=w_init,
        b_init= snt.initializers.Constant(1e-8))

    # Create additional layers if using relative positions.
    if self._relative_positions:
      self._r_k_layer = snt.Linear(
          key_proj_size,
          name='r_k_layer',
          with_bias=False,
          w_init=self._initializer)
      self._r_w_bias = tf.Variable(
          self._initializer([1, self._num_heads, 1, self._key_size],
                            dtype=tf.float32),
          name='r_w_bias')
      self._r_r_bias = tf.Variable(
          self._initializer([1, self._num_heads, 1, self._key_size],
                            dtype=tf.float32),
          name='r_r_bias')

  def _multihead_output(self, linear, inputs):
    """Applies a standard linear to inputs and returns multihead output."""

    output = snt.BatchApply(linear)(inputs)  # [B, T, H * KV]
    num_kv_channels = output.shape[-1] // self._num_heads
    # Split H * Channels into separate axes.
    output = snt.reshape(output,
                         output_shape=[-1, self._num_heads, num_kv_channels])
    # [B, T, H, KV] -> [B, H, T, KV]
    return tf.transpose(output, [0, 2, 1, 3])

  def __call__(self,
               inputs,
               is_training=False):
    # Initialise the projection layers.
    embedding_size = self._value_size * self._num_heads
    seq_len = inputs.shape[1]

    # Compute q, k and v as multi-headed projections of the inputs.
    q = self._multihead_output(self._q_layer, inputs)  # [B, H, T, K]
    k = self._multihead_output(self._k_layer, inputs)  # [B, H, T, K]
    v = self._multihead_output(self._v_layer, inputs)  # [B, H, T, V]

    # Scale the query by the square-root of key size.
    if self._scaling:
      q *= self._key_size**-0.5

    if self._relative_positions:
      # For relative positions, we project positions to form relative keys.
      distances = tf.range(-seq_len + 1, seq_len, dtype=tf.float32)[tf.newaxis]
      positional_encodings = positional_features_all(
          positions=distances,
          feature_size=self._num_relative_position_features,
          seq_length=seq_len,
          feature_functions=self._relative_position_functions,
          symmetric=self._relative_position_symmetric)
      # [1, 2T-1, Cr]

      if is_training:
        positional_encodings = tf.nn.dropout(
            positional_encodings, rate=self._positional_dropout_rate)

      # [1, H, 2T-1, K]
      r_k = self._multihead_output(self._r_k_layer, positional_encodings)

      # Add shifted relative logits to content logits.
      # [B, H, T', T]
      content_logits = tf.matmul(q + self._r_w_bias, k, transpose_b=True)
      # [B, H, T', 2T-1]
      relative_logits = tf.matmul(
          q + self._r_r_bias, r_k, transpose_b=True)
      #  [B, H, T', T]
      relative_logits = relative_shift(relative_logits)
      logits = content_logits + relative_logits
    else:
      # [B, H, T', T]
      logits = tf.matmul(q, k, transpose_b=True)

    weights = tf.nn.softmax(logits)

    # Dropout on the attention weights.
    if is_training:
      weights = tf.nn.dropout(weights, rate=self._attention_dropout_rate)

    # Transpose and reshape the output.
    output = tf.matmul(weights, v)  # [B, H, T', V]
    output_transpose = tf.transpose(output, [0, 2, 1, 3])  # [B, T', H, V]

    # Final linear layer.
    attended_inputs = snt.reshape(
        output_transpose, output_shape=[embedding_size], preserve_dims=2)
    output = self._embedding_layer(attended_inputs)

    return output

def relative_shift(x):
  """Shift the relative logits like in TransformerXL."""
  # We prepend zeros on the final timescale dimension.
  to_pad = tf.zeros_like(x[..., :1])
  x = tf.concat([to_pad, x], -1)
  _, num_heads, t1, t2 = x.shape
  x = tf.reshape(x, [-1, num_heads, t2, t1])
  x = tf.slice(x, [0, 0, 1, 0], [-1, -1, -1, -1])
  x = tf.reshape(x, [-1, num_heads, t1, t2 - 1])
  x = tf.slice(x, [0, 0, 0, 0], [-1, -1, -1, (t2 + 1) // 2])
  return x

# Available feature functions:
def get_positional_feature_function(name):
  """Returns positional feature functions."""
  available = {
      'positional_features_exponential': positional_features_exponential,
      'positional_features_central_mask': positional_features_central_mask,
      'positional_features_gamma': positional_features_gamma
  }
  if name not in available:
    raise ValueError(f'Function {name} not available in {available.keys()}')
  return available[name]


def positional_features_all(positions: tf.Tensor,
                            feature_size: int,
                            seq_length: Optional[int] = None,
                            bin_size: Optional[int] = None,
                            feature_functions: Optional[List[str]] = None,
                            symmetric=False):
  """Compute relative positional encodings/features.

  Each positional feature function will compute/provide the same fraction of
  features, making up the total of feature_size.

  Args:
    positions: Tensor of relative positions of arbitrary shape.
    feature_size: Total number of basis functions.
    seq_length: Sequence length denoting the characteristic length that
      the individual positional features can use. This is required since the
      parametrization of the input features should be independent of `positions`
      while it could still require to use the total number of features.
    bin_size: Bin sized used to partition the sequence. This can be used to
      compute features on the absolute scale relative to the genome.
    feature_functions: List of different feature functions to use. Each function
      will take as argument: positions, sequence length and number of features
      to compute.
    symmetric: If True, the resulting features will be symmetric across the
      relative position of 0 (i.e. only absolute value of positions will
      matter). If false, then both the symmetric and asymmetric version
      (symmetric multiplied by sign(positions)) of the features will be used.

  Returns:
    Tensor of shape: `positions.shape + (feature_size,)`.
  """
  if feature_functions is None:
    feature_functions = ['positional_features_exponential',
                         'positional_features_central_mask',
                         'positional_features_gamma']
  num_components = len(feature_functions)  # 1 per each basis function
  if not symmetric:
    num_components = 2 * num_components

  # For now, we do not allow odd sized embeddings.
  if feature_size % num_components != 0:
    raise ValueError(
        f'feature_size has to be divisible by {num_components}')

  feature_functions = [get_positional_feature_function(f)
                       for f in feature_functions]
  num_basis_per_class = feature_size // num_components
  embeddings = tf.concat([f(tf.abs(positions), num_basis_per_class,
                            seq_length, bin_size)
                          for f in feature_functions],
                         axis=-1)
  if not symmetric:
    embeddings = tf.concat([embeddings,
                            tf.sign(positions)[..., tf.newaxis] * embeddings],
                           axis=-1)
  tf.TensorShape(embeddings.shape).assert_is_compatible_with(
      positions.shape + [feature_size])
  return embeddings


def _prepend_dims(x, num_dims):
  return tf.reshape(x, shape=[1] * num_dims + x.shape)


def positional_features_exponential(positions: tf.Tensor,
                                    feature_size: int,
                                    seq_length: Optional[int] = None,
                                    bin_size: Optional[int] = None,
                                    min_half_life: Optional[float] = 3.0):
  """Create exponentially decaying positional weights.

  Args:
    positions: Position tensor (arbitrary shape).
    feature_size: Number of basis functions to use.
    seq_length: Sequence length.
    bin_size: (unused). See `positional_features_all`.
    min_half_life: Smallest exponential half life in the grid of half lives.

  Returns:
    A Tensor with shape [2 * seq_length - 1, feature_size].
  """
  del bin_size  # Unused.
  if seq_length is None:
    seq_length = tf.reduce_max(tf.abs(positions)) + 1
  # Grid of half lifes from [3, seq_length / 2] with feature_size
  # distributed on the log scale.
  seq_length = tf.cast(seq_length, dtype=tf.float32)
  max_range = tf.math.log(seq_length) / tf.math.log(2.0)
  half_life = tf.pow(2.0, tf.linspace(min_half_life, max_range, feature_size))
  half_life = _prepend_dims(half_life, positions.shape.rank)
  positions = tf.abs(positions)
  outputs = tf.exp(-tf.math.log(2.0) / half_life * positions[..., tf.newaxis])
  tf.TensorShape(outputs.shape).assert_is_compatible_with(
      positions.shape + [feature_size])
  return outputs


def positional_features_central_mask(positions: tf.Tensor,
                                     feature_size: int,
                                     seq_length: Optional[int] = None,
                                     bin_size: Optional[int] = None):
  """Positional features using a central mask (allow only central features)."""
  del seq_length  # Unused.
  del bin_size  # Unused.
  center_widths = tf.pow(2.0, tf.range(1, feature_size + 1, dtype=tf.float32))
  center_widths = center_widths - 1
  center_widths = _prepend_dims(center_widths, positions.shape.rank)
  outputs = tf.cast(center_widths > tf.abs(positions)[..., tf.newaxis],
                    tf.float32)
  tf.TensorShape(outputs.shape).assert_is_compatible_with(
      positions.shape + [feature_size])
  return outputs


def gamma_pdf(x, concentration, rate):
  """Gamma probability distribution function: p(x|concentration, rate)."""
  log_unnormalized_prob = tf.math.xlogy(concentration - 1., x) - rate * x
  log_normalization = (tf.math.lgamma(concentration) -
                       concentration * tf.math.log(rate))
  return tf.exp(log_unnormalized_prob - log_normalization)


def positional_features_gamma(positions: tf.Tensor,
                              feature_size: int,
                              seq_length: Optional[int] = None,
                              bin_size: Optional[int] = None,
                              stddev=None,
                              start_mean=None):
  """Positional features computed using the gamma distributions."""
  del bin_size  # Unused.
  if seq_length is None:
    seq_length = tf.reduce_max(tf.abs(positions)) + 1
  if stddev is None:
    stddev = seq_length / (2 * feature_size)
  if start_mean is None:
    start_mean = seq_length / feature_size
  mean = tf.linspace(start_mean, seq_length, num=feature_size)
  mean = _prepend_dims(mean, positions.shape.rank)
  concentration = (mean / stddev)**2
  rate = mean / stddev**2
  probabilities = gamma_pdf(
      tf.abs(tf.cast(positions, dtype=tf.float32))[..., tf.newaxis],
      concentration, rate)
  probabilities += 1e-8  # To ensure numerical stability.
  outputs = probabilities / tf.reduce_max(probabilities)
  tf.TensorShape(outputs.shape).assert_is_compatible_with(
      positions.shape + [feature_size])
  return outputs

class Enformer(snt.Module):
  """Main model."""

  def __init__(self,
               channels: int = 1536,
               num_transformer_layers: int = 11,
               num_heads: int = 8,
               pooling_type: str = 'attention',
               name: str = 'enformer'):
    """Enformer model.

    Args:
      channels: Number of convolutional filters and the overall 'width' of the
        model.
      num_transformer_layers: Number of transformer layers.
      num_heads: Number of attention heads.
      pooling_type: Which pooling function to use. Options: 'attention' or max'.
      name: Name of sonnet module.
    """
    super().__init__(name=name)
    # pylint: disable=g-complex-comprehension,g-long-lambda,cell-var-from-loop
    heads_channels = {'human': 5313, 'mouse': 1643}
    dropout_rate = 0.4
    assert channels % num_heads == 0, ('channels needs to be divisible '
                                       f'by {num_heads}')
    whole_attention_kwargs = {
        'attention_dropout_rate': 0.05,
        'initializer': None,
        'key_size': 64,
        'num_heads': num_heads,
        'num_relative_position_features': channels // num_heads,
        'positional_dropout_rate': 0.01,
        'relative_position_functions': [
            'positional_features_exponential',
            'positional_features_central_mask',
            'positional_features_gamma'
        ],
        'relative_positions': True,
        'scaling': True,
        'value_size': channels // num_heads,
        'zero_initialize': True
    }

    trunk_name_scope = tf.name_scope('trunk')
    trunk_name_scope.__enter__()
    from sonnet.src import moving_averages

    # lambda is used in Sequential to construct the module under tf.name_scope.
    def conv_block(filters, width=1, w_init=None, name='conv_block', **kwargs):
      with tf.name_scope(name or "batch_norm"):
        moving_mean = moving_averages.ExponentialMovingAverage(
            0.9, name="moving_mean")
        moving_variance = moving_averages.ExponentialMovingAverage(
            0.9, name="moving_variance")
      return Sequential(lambda: [
          snt.distribute.CrossReplicaBatchNorm(create_scale=True,
                        create_offset=True,
                        moving_mean = moving_mean,
                        moving_variance = moving_variance,
                        scale_init=snt.initializers.Ones()),
          gelu,
          snt.Conv1D(filters, width, w_init=w_init, **kwargs)
      ], name=name)

    stem = Sequential(lambda: [
        snt.Conv1D(channels // 2, 15),
        Residual(conv_block(channels // 2, 1, name='pointwise_conv_block')),
        pooling_module(pooling_type, pool_size=2),
    ], name='stem')

    filter_list = exponential_linspace_int(start=channels // 2, end=channels,
                                           num=6, divisible_by=128)
    conv_tower = Sequential(lambda: [
        Sequential(lambda: [
            conv_block(num_filters, 5),
            Residual(conv_block(num_filters, 1, name='pointwise_conv_block')),
            pooling_module(pooling_type, pool_size=2),
            ],
                   name=f'conv_tower_block_{i}')
        for i, num_filters in enumerate(filter_list)], name='conv_tower')

    # Transformer.
    def transformer_mlp():
      return Sequential(lambda: [
          snt.LayerNorm(axis=-1, create_scale=True, create_offset=True),
          snt.Linear(channels * 2, name = 'project_in'),
          snt.Dropout(dropout_rate),
          tf.nn.relu,
          snt.Linear(channels, name = 'project_out'),
          snt.Dropout(dropout_rate)], name='mlp')

    transformer = Sequential(lambda: [
        Sequential(lambda: [
            Residual(Sequential(lambda: [
                snt.LayerNorm(axis=-1,
                              create_scale=True, create_offset=True,
                              scale_init=snt.initializers.Ones()),
                MultiheadAttention(**whole_attention_kwargs,
                                                    name=f'attention_{i}'),
                snt.Dropout(dropout_rate),
            ], name='mha')),
            Residual(transformer_mlp())], name=f'transformer_block_{i}')
        for i in range(num_transformer_layers)], name='transformer')

    crop_final = TargetLengthCrop1D(TARGET_LENGTH, name='target_input')

    final_pointwise = Sequential(lambda: [
        conv_block(channels * 2, 1),
        snt.Dropout(dropout_rate / 8),
        gelu], name='final_pointwise')

    self._trunk = Sequential([stem,
                              conv_tower,
                              transformer,
                              crop_final,
                              final_pointwise],
                             name='trunk')
    trunk_name_scope.__exit__(None, None, None)

    with tf.name_scope('heads'):
      self._heads = {
          head: Sequential(
              lambda: [snt.Linear(num_channels), tf.nn.softplus],
              name=f'head_{head}')
          for head, num_channels in heads_channels.items()
      }
    # pylint: enable=g-complex-comprehension,g-long-lambda,cell-var-from-loop

  @property
  def trunk(self):
    return self._trunk

  @property
  def heads(self):
    return self._heads

  def __call__(self, inputs: tf.Tensor,
               is_training: bool) -> Dict[str, tf.Tensor]:
    trunk_embedding = self.trunk(inputs, is_training=is_training)
    return {
        head: head_module(trunk_embedding, is_training=is_training)
        for head, head_module in self.heads.items()
    }

  @tf.function(input_signature=[
      tf.TensorSpec([None, SEQUENCE_LENGTH, 4], tf.float32)])
  def predict_on_batch(self, x):
    """Method for SavedModel."""
    return self(x, is_training=False)


class TargetLengthCrop1D(snt.Module):
  """Crop sequence to match the desired target length."""

  def __init__(self, target_length: int, name='target_length_crop'):
    super().__init__(name=name)
    self._target_length = target_length

  def __call__(self, inputs):
    trim = (inputs.shape[-2] - self._target_length) // 2
    if trim < 0:
      raise ValueError('inputs longer than target length')

    return inputs[..., trim:-trim, :]


class Sequential(snt.Module):
  """snt.Sequential automatically passing is_training where it exists."""

  def __init__(self,
               layers: Optional[Union[Callable[[], Iterable[snt.Module]],
                                      Iterable[Callable[..., Any]]]] = None,
               name: Optional[Text] = None):
    super().__init__(name=name)
    if layers is None:
      self._layers = []
    else:
      # layers wrapped in a lambda function to have a common namespace.
      if hasattr(layers, '__call__'):
        with tf.name_scope(name):
          layers = layers()
      self._layers = [layer for layer in layers if layer is not None]

  def __call__(self, inputs: tf.Tensor, is_training: bool, **kwargs):
    outputs = inputs
    for _, mod in enumerate(self._layers):
      if accepts_is_training(mod):
        outputs = mod(outputs, is_training=is_training, **kwargs)
      else:
        outputs = mod(outputs, **kwargs)
    return outputs


def pooling_module(kind, pool_size):
  """Pooling module wrapper."""
  if kind == 'attention':
    return SoftmaxPooling1D(pool_size=pool_size, per_channel=True,
                            w_init_scale=2.0)
  elif kind == 'max':
    return tf.keras.layers.MaxPool1D(pool_size=pool_size, padding='same')
  else:
    raise ValueError(f'Invalid pooling kind: {kind}.')


class SoftmaxPooling1D(snt.Module):
  """Pooling operation with optional weights."""

  def __init__(self,
               pool_size: int = 2,
               per_channel: bool = False,
               w_init_scale: float = 0.0,
               name: str = 'softmax_pooling'):
    """Softmax pooling.

    Args:
      pool_size: Pooling size, same as in Max/AvgPooling.
      per_channel: If True, the logits/softmax weights will be computed for
        each channel separately. If False, same weights will be used across all
        channels.
      w_init_scale: When 0.0 is equivalent to avg pooling, and when
        ~2.0 and `per_channel=False` it's equivalent to max pooling.
      name: Module name.
    """
    super().__init__(name=name)
    self._pool_size = pool_size
    self._per_channel = per_channel
    self._w_init_scale = w_init_scale
    self._logit_linear = None

  @snt.once
  def _initialize(self, num_features):
    self._logit_linear = snt.Linear(
        output_size=num_features if self._per_channel else 1,
        with_bias=False,  # Softmax is agnostic to shifts.
        w_init=snt.initializers.Identity(self._w_init_scale))

  def __call__(self, inputs):
    _, length, num_features = inputs.shape
    self._initialize(num_features)
    inputs = tf.reshape(
        inputs,
        (-1, length // self._pool_size, self._pool_size, num_features))
    return tf.reduce_sum(
        inputs * tf.nn.softmax(self._logit_linear(inputs), axis=-2),
        axis=-2)


class Residual(snt.Module):
  """Residual block."""

  def __init__(self, module: snt.Module, name='residual'):
    super().__init__(name=name)
    self._module = module

  def __call__(self, inputs: tf.Tensor, is_training: bool, *args,
               **kwargs) -> tf.Tensor:
    return inputs + self._module(inputs, is_training, *args, **kwargs)


def gelu(x: tf.Tensor) -> tf.Tensor:
  """Applies the Gaussian error linear unit (GELU) activation function.

  Using approximiation in section 2 of the original paper:
  https://arxiv.org/abs/1606.08415

  Args:
    x: Input tensor to apply gelu activation.
  Returns:
    Tensor with gelu activation applied to it.
  """
  return tf.nn.sigmoid(1.702 * x) * x


def one_hot_encode(sequence: str,
                   alphabet: str = 'ACGT',
                   neutral_alphabet: str = 'N',
                   neutral_value: Any = 0,
                   dtype=np.float32) -> np.ndarray:
  """One-hot encode sequence."""
  def to_uint8(string):
    return np.frombuffer(string.encode('ascii'), dtype=np.uint8)
  hash_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
  hash_table[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
  hash_table[to_uint8(neutral_alphabet)] = neutral_value
  hash_table = hash_table.astype(dtype)
  return hash_table[to_uint8(sequence)]


def exponential_linspace_int(start, end, num, divisible_by=1):
  """Exponentially increasing values of integers."""
  def _round(x):
    return int(np.round(x / divisible_by) * divisible_by)

  base = np.exp(np.log(end / start) / (num - 1))
  return [_round(start * base**i) for i in range(num)]


def accepts_is_training(module):
  return 'is_training' in list(inspect.signature(module.__call__).parameters)

# data related functions

# @title `get_targets(organism)`
def get_targets(organism):
  targets_txt = f'https://raw.githubusercontent.com/calico/basenji/master/manuscripts/cross2020/targets_{organism}.txt'
  return pd.read_csv(targets_txt, sep='\t')

# @title `get_dataset(organism, subset, num_threads=8)`

def reverse_complement_transform(seq):
  """Reverse complement of batched onehot seq and corresponding label and na."""

  # reverse complement sequence
  seq_rc = tf.gather(seq, [3, 2, 1, 0], axis=-1)
  seq_rc = tf.reverse(seq_rc, axis=[0])
  return seq_rc


def shift_sequence(seq, shift_amount, pad_value=0.25):
  """Shift a sequence left or right by shift_amount.
  Args:
    seq: a [batch_size, sequence_length, sequence_depth] sequence to shift
    shift_amount: the signed amount to shift (tf.int32 or int)
    pad_value: value to fill the padding (primitive or scalar tf.Tensor)
  """
  input_shape = seq.shape

  pad = pad_value * tf.ones_like(seq[0:tf.abs(shift_amount), :])

  def _shift_right(_seq):
    sliced_seq = _seq[:-shift_amount:, :]
    return tf.concat([pad, sliced_seq], axis=0)

  def _shift_left(_seq):
    sliced_seq = _seq[-shift_amount:, :]
    return tf.concat([sliced_seq, pad], axis=0)

  output = tf.cond(
      tf.greater(shift_amount, 0), lambda: _shift_right(seq),
      lambda: _shift_left(seq))

  output.set_shape(input_shape)
  return output

def augment_stochastic_shifts(seq, augment_shifts):
  """Apply a stochastic shift augmentation.
  Args:
    seq: input sequence of size [batch_size, length, depth]
    augment_shifts: list of int offsets to sample from
  Returns:
    shifted and padded sequence of size [batch_size, length, depth]
  """
  shift_index = tf.random.uniform(shape=[], minval=0,
      maxval=len(augment_shifts), dtype=tf.int64)
  shift_value = tf.gather(tf.constant(augment_shifts), shift_index)

  seq = tf.cond(tf.not_equal(shift_value, 0),
                lambda: shift_sequence(seq, shift_value),
                lambda: seq)

  return seq

def augment_stochastic_shifts_map_fn(datum):
  augment_shifts = [-2, -1, 0, 1, 2]
  return dict(
    sequence = augment_stochastic_shifts(datum['sequence'], augment_shifts),
    target = datum['target']
  )

def augment_stochastic_rc_map_fn(datum):
  sequence, target = (datum['sequence'], datum['target'])
  augment = tf.random.uniform(shape=[]) > 0.5
  sequence = tf.cond(augment, lambda: reverse_complement_transform(sequence),
                              lambda: sequence)
  target = tf.cond(augment, lambda: tf.reverse(target, axis=[0]),
                            lambda: target)
  return dict(sequence = sequence, target = target)


def organism_path(organism):
    return os.path.join(f'gs://basenji_barnyard/data', organism)


def get_dataset(organism, subset, num_threads=8, shuffle=True, rotate = 0, augment = False):
  metadata = get_metadata(organism)
  files = tfrecord_files(organism, subset) 
  files = files[rotate:] + files[:rotate]
  dataset = tf.data.TFRecordDataset(files,
                                    compression_type='ZLIB',
                                    num_parallel_reads=num_threads)
  if shuffle:
    dataset = dataset.repeat()
    dataset = dataset.shuffle(5000, seed = 42)

  dataset = dataset.map(functools.partial(deserialize, metadata=metadata),
                        num_parallel_calls=num_threads)
  if augment:
    dataset = dataset.map(augment_stochastic_shifts_map_fn, num_parallel_calls=num_threads)
    dataset = dataset.map(augment_stochastic_rc_map_fn, num_parallel_calls=num_threads)

  return dataset


def get_metadata(organism):
  # Keys:
  # num_targets, train_seqs, valid_seqs, test_seqs, seq_length,
  # pool_width, crop_bp, target_length
  path = os.path.join(organism_path(organism), 'statistics.json')
  with tf.io.gfile.GFile(path, 'r') as f:
    return json.load(f)


def tfrecord_files(organism, subset):
  # Sort the values by int(*).
  return sorted(tf.io.gfile.glob(os.path.join(
      organism_path(organism), 'tfrecords', f'{subset}-*.tfr'
  )), key=lambda x: int(x.split('-')[-1].split('.')[0]))


def deserialize(serialized_example, metadata):
  """Deserialize bytes stored in TFRecordFile."""
  feature_map = {
      'sequence': tf.io.FixedLenFeature([], tf.string),
      'target': tf.io.FixedLenFeature([], tf.string),
  }
  example = tf.io.parse_example(serialized_example, feature_map)
  sequence = tf.io.decode_raw(example['sequence'], tf.bool)
  sequence = tf.reshape(sequence, (metadata['seq_length'], 4))
  sequence = tf.cast(sequence, tf.float32)

  target = tf.io.decode_raw(example['target'], tf.float16)
  target = tf.reshape(target,
                      (metadata['target_length'], metadata['num_targets']))
  target = tf.cast(target, tf.float32)

  return {'sequence': sequence,
          'target': target}

# training related functions

def corr_coef(x, y):
  x2 = tf.math.square(x)
  y2 = tf.math.square(y)
  xy = x * y
  ex = tf.reduce_mean(x, axis = 1)
  ey = tf.reduce_mean(y, axis = 1)
  exy = tf.reduce_mean(xy, axis = 1)
  ex2 = tf.reduce_mean(x2, axis = 1)
  ey2 = tf.reduce_mean(y2, axis = 1)
  r = (exy - ex * ey) / ((tf.math.sqrt(ex2 - tf.math.square(ex)) * tf.math.sqrt(ey2 - tf.math.square(ey))) + 1e-8)
  return tf.reduce_mean(r, axis = -1)

def create_eval_step(model, head):
  @tf.function
  def predict(seq, target):
    pred = model(seq, is_training=False)[head]
    return corr_coef(pred, target)
  return predict

def create_step_function(model, optimizer, head, clip_grad_norm = 1.0, weight_decay = 0.0001):

  @tf.function
  def train_step(batch_seq, batch_target):
    with tf.GradientTape() as tape:
      outputs = model(batch_seq, is_training=True)[head]
      corr_coef_loss = 1 - corr_coef(outputs, batch_target) 
      poisson = tf.reduce_mean(
          tf.keras.losses.poisson(batch_target, outputs))
      loss = poisson

    gradients = tape.gradient(loss, model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    gradients = [tf.clip_by_norm(grad, clip_grad_norm) for grad in gradients]
    ctx = tf.distribute.get_replica_context()
    gradients = ctx.all_reduce("mean", gradients)
    optimizer.apply(gradients, model.trainable_variables)
    return loss

  return train_step

# instantiate model and training / eval functions

with tpu_strategy.scope():
  model = Enformer(channels=1536,
                   num_heads=8,
                   num_transformer_layers=11)

  learning_rate = tf.Variable(0., trainable=False, name='learning_rate')
  optimizer = snt.optimizers.Adam(learning_rate=learning_rate)

  train_step_human = create_step_function(model, optimizer, 'human')
  train_step_mouse = create_step_function(model, optimizer, 'mouse')

  eval_step_human = create_eval_step(model, 'human')
  eval_step_mouse = create_eval_step(model, 'mouse')

# experiment tracker

wandb.init(project='enformer')
wandb.run.save()

# Train the model

num_steps = int(2e6)
num_warmup_steps = 5000
target_learning_rate = 5e-4

checkpoint_every = 2500
max_eval_steps = 25
eval_every = 500

# Step variables

global_step = tf.Variable(0, name='global_step', trainable=False)

# checkpointing

checkpoint_root = "gs://enformer/"
checkpoint_name = "enformer"

save_prefix = os.path.join(checkpoint_root, checkpoint_name)

checkpoint = tf.train.Checkpoint(module = model,  step = global_step, optimizer = optimizer)

# load latest checkpoint if possible

latest = tf.train.latest_checkpoint(checkpoint_root)
if latest is not None:
  checkpoint.restore(latest)

@tf.function
def step():
  global_step.assign(global_step + 1)

  batch_human, batch_mouse = next(data_it)
  loss_human = tpu_strategy.run(train_step_human, args = (batch_human['sequence'], batch_human['target']))
  loss_mouse = tpu_strategy.run(train_step_mouse, args = (batch_mouse['sequence'], batch_mouse['target']))

  loss_human = tpu_strategy.reduce('mean', loss_human, axis = None)
  loss_mouse = tpu_strategy.reduce('mean', loss_mouse, axis = None)

  learning_rate_frac = tf.math.minimum(1.0, tf.cast(global_step, tf.float32) / tf.math.maximum(1.0, float(num_warmup_steps)))      
  learning_rate.assign(target_learning_rate * learning_rate_frac)

  return loss_human, loss_mouse

@tf.function
def eval_step():
  batch_human = next(valid_human_data_it)
  batch_mouse = next(valid_mouse_data_it)
  human_r = tpu_strategy.run(eval_step_human, args = (batch_human['sequence'], batch_human['target']))
  mouse_r = tpu_strategy.run(eval_step_mouse, args = (batch_mouse['sequence'], batch_mouse['target']))

  human_r = tpu_strategy.reduce('mean', human_r, axis = 0)
  mouse_r = tpu_strategy.reduce('mean', mouse_r, axis = 0)
  return human_r, mouse_r

i = global_step.numpy()

total_mice = 114 * 256 + 111
total_human = 132 * 256 + 229
bucket_size = 256
num_seen = i * num_cores
human_file_skip = (num_seen % total_human) // bucket_size
mouse_file_skip = (num_seen % total_mice) // bucket_size

human_dataset = get_dataset('human', 'train', rotate = human_file_skip).batch(num_cores, drop_remainder = True)
mouse_dataset = get_dataset('mouse', 'train', rotate = mouse_file_skip).batch(num_cores, drop_remainder = True)
human_mouse_dataset = tf.data.Dataset.zip((human_dataset, mouse_dataset)).prefetch(2)

human_valid_dataset = get_dataset('human', 'valid', shuffle = False).repeat().batch(num_cores)
mouse_valid_dataset = get_dataset('mouse', 'valid', shuffle = False).repeat().batch(num_cores)

data_it = iter(tpu_strategy.experimental_distribute_dataset(human_mouse_dataset))
valid_human_data_it = iter(tpu_strategy.experimental_distribute_dataset(human_valid_dataset))
valid_mouse_data_it = iter(tpu_strategy.experimental_distribute_dataset(mouse_valid_dataset))

print(f'starting from {i}')

while i < num_steps:
  print(f'processing step {i}')
  loss_human, loss_mouse = step()
  loss_human = loss_human.numpy()
  loss_mouse = loss_mouse.numpy()
  learning_rate_numpy = learning_rate.numpy()
  print(f'completed step {i}')
  log = {
    'loss_human': loss_human,
    'loss_mouse': loss_mouse,
    'learning_rate': learning_rate_numpy
  }

  if i and not i % eval_every:
    print('evaluating')

    human_pearson_r, mouse_pearson_r = eval_step()
    human_pearson_r = human_pearson_r.numpy()
    mouse_pearson_r = mouse_pearson_r.numpy()

    log = {
      **log,
      'human_pearson_r': human_pearson_r,
      'mouse_pearson_r': mouse_pearson_r
    }

  wandb.log(log, step = i)

  if not i % checkpoint_every:
    print('checkpointing')
    checkpoint.save(save_prefix)

  i += 1
