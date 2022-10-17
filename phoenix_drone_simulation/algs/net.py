""" Implementation of custom network classes for RL algorithms.

Author:     Sven Gronauer (sven.gronauer@tum.de), Daniel Stümke (daniel.stuemke@gmail.com)
based on:   Spinning Up's Vanilla Policy Gradient
            https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/vpg/core.py
"""
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf

# Disable tensorflow GPU
tf.config.set_visible_devices([], 'GPU')

def initialize_layer(
        init_function: str,
        layer: torch.nn.Module
):
    if init_function == 'kaiming_uniform':  # this the default!
        nn.init.kaiming_uniform_(layer.weight, a=np.sqrt(5))
    elif init_function == 'xavier_normal':
        nn.init.xavier_normal_(layer.weight)
    # glorot is also known as xavier uniform
    elif init_function == 'glorot' or init_function == 'xavier_uniform':
        nn.init.xavier_uniform_(layer.weight)
    elif init_function == 'orthogonal':  # matches values from baselines repo.
        nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
    else:
        raise NotImplementedError

def convert_str_to_torch_functional(activation):
    if isinstance(activation, str):  # convert string to torch functional
        activations = {
            'identity': nn.Identity,
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'softplus': nn.Softplus,
            'tanh': nn.Tanh
        }
        assert activation in activations
        activation = activations[activation]
    assert issubclass(activation, torch.nn.Module)
    return activation

def convert_str_to_tf_functional(activation):
    if isinstance(activation, str):  # convert string to tensorflow functional
        activations = {
            'identity': lambda **kwargs: tf.keras.layers.Activation('linear',**kwargs),
            'relu': lambda **kwargs: tf.keras.layers.Activation('relu', **kwargs),
            'sigmoid': lambda **kwargs: tf.keras.layers.Activation('sigmoid', **kwargs),
            'softplus': lambda **kwargs: tf.keras.layers.Activation('softplus', **kwargs),
            'tanh': lambda **kwargs: tf.keras.layers.Activation('tanh', **kwargs)
        }
        assert activation in activations
        activation = activations[activation]
    return activation

def convert_str_to_tf_layer(layer):
    if isinstance(layer, str):  # convert string to tensorflow layer
        layers = {
            'GRU': tf.keras.layers.GRU,
            'LSTM': tf.keras.layers.LSTM
        }
        assert layer in layers
        layer = layers[layer]
    return layer    

def convert_str_to_torch_layer(layer):
    if isinstance(layer, str):  # convert string to torch layer
        layers = {
            'GRU': nn.GRU,
            'LSTM': nn.LSTM
        }
        assert layer in layers
        layer = layers[layer]
    assert issubclass(layer, torch.nn.Module)
    return layer

class StatefulRNN(nn.Module):
  def __init__(self, layer):
    super().__init__()
    self.layer = layer
    self.state = None

  def reset_states(self):
    self.state = None

  def forward(self, x):
    inp_size = x.size()
    if len(inp_size) < 3:
        # Bring data into Format: [Batch,Seq,Feat]
        x = x.view(1,-1,x.size()[-1])
    y, state = self.layer(x, self.state)
    self.state = self._detach_state(state)
    y_size = list(inp_size)
    y_size[-1] = y.size()[-1]
    y = y.view(y_size)
    return y

  def _detach_state(self, state):
    # Detach hidden state from gradient computation and replace nan's with 0
    if isinstance(state, tuple):
      return tuple(torch.nan_to_num(s.detach()) for s in state)
    if isinstance(state, list):
      return [torch.nan_to_num(s.detach()) for s in state]
    return torch.nan_to_num(state.detach())

def build_recurrent_network(
        sizes,
        activation='identity',
        output_activation='identity',
        weight_initialization='kaiming_uniform',
        layer='GRU'
):
    tf_model = build_recurrent_tf(**locals())
    layer = convert_str_to_torch_layer(layer)
    activation = convert_str_to_torch_functional(activation)
    output_activation = convert_str_to_torch_functional(output_activation)
    layers = list()
    layers_rnn = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        lay_stateless = layer(sizes[j], sizes[j + 1], batch_first=True) if j < len(sizes) - 2 else None
        if lay_stateless is None:
            lay_affine = nn.Linear(sizes[j], sizes[j + 1])
            initialize_layer(weight_initialization, lay_affine)
            lay_statefull = lay_affine
        else:
            lay_statefull = StatefulRNN(lay_stateless)
            layers_rnn.append(lay_statefull)
        layers += [lay_statefull, act()]
    return nn.Sequential(*layers), layers_rnn, tf_model

def build_recurrent_tf(
        sizes,
        activation='identity',
        output_activation='identity',
        weight_initialization='kaiming_uniform',
        layer='GRU',
        **kwargs
      ):
    layer = convert_str_to_tf_layer(layer)
    activation = convert_str_to_tf_functional(activation)
    output_activation = convert_str_to_tf_functional(output_activation)
    inp = tf.keras.Input(shape=(1,sizes[0]), batch_size=1)
    x = tf.keras.layers.BatchNormalization(center=False, scale=False, epsilon=1e-10)(inp)
    for j in range(1,len(sizes)):
        act = activation() if j < len(sizes) - 1 else output_activation()
        lay = layer(sizes[j], return_sequences=True, stateful=True, name=f"hidden{j}") \
            if j < len(sizes) - 1 else tf.keras.layers.Dense(sizes[j], name="output")
        x = lay(x)
        x = act(x)
    return tf.keras.Model(inp,x)

class CascadedNN(nn.Module):
  def __init__(self, layer_out, layer_in1, layer_in2=None):
    super().__init__()
    self.layer_in1 = layer_in1
    self.layer_in2 = layer_in2
    self.layer_out = layer_out

  def forward(self, x):
    y_1 = self.layer_in1(x)
    if self.layer_in2 is None:
        y_2 = x
    else:
        y_2 = self.layer_in2(x)
    x_12 = torch.cat((y_1,y_2), -1)
    y = self.layer_out(x_12)
    return y

def build_cascaded_network(
        sizes,
        activation,
        output_activation='identity',
        weight_initialization='kaiming_uniform',
        layer='GRU'
):
    tf_model = build_cascaded_tf(**locals())
    layer = convert_str_to_torch_layer(layer)
    activation = convert_str_to_torch_functional(activation)
    output_activation = convert_str_to_torch_functional(output_activation)
    net = None
    layers_rnn = []
    for j in range((len(sizes) - 2)//2):
        lay_rnn = StatefulRNN(layer(sizes[j*2], sizes[j*2 + 1], batch_first=True))
        layers_rnn.append(lay_rnn)
        lay_lin = nn.Linear(sizes[j*2 + 1] + sizes[j*2], sizes[(j + 1)*2])
        initialize_layer(weight_initialization, lay_lin)
        net_pre = net
        net = nn.Sequential(CascadedNN(lay_lin, lay_rnn, net_pre), activation())
    return (
        nn.Sequential(net, nn.Linear(sizes[-2], sizes[-1]), output_activation()),
        layers_rnn, tf_model )

def build_cascaded_tf(
        sizes,
        activation,
        output_activation='identity',
        weight_initialization='kaiming_uniform',
        layer='GRU',
        **kwargs
      ):
    layer = convert_str_to_tf_layer(layer)
    activation = convert_str_to_tf_functional(activation)
    output_activation = convert_str_to_tf_functional(output_activation)
    inp = tf.keras.Input(shape=(1,sizes[0]), batch_size=1)
    x = tf.keras.layers.BatchNormalization(center=False, scale=False, epsilon=1e-10)(inp)
    net = None
    for j in range((len(sizes) - 2)//2):
        lay_rnn = layer(sizes[j*2 + 1], return_sequences=True, stateful=True, name=f"hidden{j*2 + 1}")
        lay_lin = tf.keras.layers.Dense(sizes[(j + 1)*2], name=f"hidden{(j + 1)*2}")
        net_pre = net if net is not None else x
        net = lay_rnn(net_pre)
        net = lay_lin(tf.keras.layers.Concatenate(name=f"concat{j*2+1}n{(j + 1)*2}")([
            net, net_pre
        ]))
        net = activation()(net)
    net = tf.keras.layers.Dense(sizes[-1], name="output")(net)
    net = output_activation()(net)
    return tf.keras.Model(inp, net)



def build_forward_network(
        sizes,
        activation,
        output_activation='identity',
        weight_initialization='kaiming_uniform'
):
    tf_model = build_forward_tf(**locals())
    activation = convert_str_to_torch_functional(activation)
    output_activation = convert_str_to_torch_functional(output_activation)
    layers = list()
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        affine_layer = nn.Linear(sizes[j], sizes[j + 1])
        initialize_layer(weight_initialization, affine_layer)
        layers += [affine_layer, act()]
    return nn.Sequential(*layers), tf_model

def build_forward_tf(
        sizes,
        activation,
        output_activation='identity',
        weight_initialization='kaiming_uniform',
        **kwargs
      ):
    layer = tf.keras.layers.Dense
    activation = convert_str_to_tf_functional(activation)
    output_activation = convert_str_to_tf_functional(output_activation)
    inp = tf.keras.Input(shape=(1,sizes[0]), batch_size=1)
    x = tf.keras.layers.BatchNormalization(center=False, scale=False, epsilon=1e-10)(inp)
    for j in range(1,len(sizes)):
        act = activation() if j < len(sizes) - 1 else output_activation()
        lay = layer(sizes[j], name=f"hidden{j}") \
            if j < len(sizes) - 1 else layer(sizes[j], name="output")
        x = lay(x)
        x = act(x)
    return tf.keras.Model(inp,x)