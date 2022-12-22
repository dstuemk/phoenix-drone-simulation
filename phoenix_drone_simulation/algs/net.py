""" Implementation of custom network classes for RL algorithms.

Author:     Sven Gronauer (sven.gronauer@tum.de), Daniel Stümke (daniel.stuemke@gmail.com)
based on:   Spinning Up's Vanilla Policy Gradient
            https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/vpg/core.py
"""
import numpy as np
import torch
import torch.nn as nn

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

def convert_str_to_torch_layer(layer):
    if isinstance(layer, str):  # convert string to torch layer
        layers = {
            'GRU':  lambda *args,**kwargs: nn.GRU(*args,**kwargs,batch_first=True),
            'LSTM': lambda *args,**kwargs: nn.LSTM(*args,**kwargs,batch_first=True),
            'FC':   nn.Linear,
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

def initialize_weights(layer, initialization):
    if initialization is None:
        return layer
    init_funcs = {
        'kaiming_uniform': lambda : nn.init.kaiming_uniform_(layer.weight, a=np.sqrt(5)),
        'xavier_normal':   lambda : nn.init.xavier_normal_(layer.weight),
        'glorot':          lambda : nn.init.xavier_uniform_(layer.weight),
        'xavier_uniform':  lambda : nn.init.xavier_uniform_(layer.weight),
        'orthogonal':      lambda : nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
    }
    if not initialization in init_funcs:
        raise NotImplementedError()
    init_fun = init_funcs[initialization]
    init_fun()
    return layer

def get_layer(type, initialization):
    layers = {
        'GRU':  lambda *args,**kwargs: initialize_weights(
            StatefulRNN(nn.GRU(*args,**kwargs,batch_first=True)),
            initialization),
        'LSTM': lambda *args,**kwargs: initialize_weights(
            StatefulRNN(nn.LSTM(*args,**kwargs,batch_first=True)),
            initialization),
        'FC':   lambda *args,**kwargs: initialize_weights(
            nn.Linear(*args,**kwargs),
            initialization),
    }
    if not type in layers:
        raise NotImplementedError()
    return layers[type]

def get_activation(activation):
    activations = {
        'identity': nn.Identity,
        'relu': nn.ReLU,
        'sigmoid': nn.Sigmoid,
        'softplus': nn.Softplus,
        'tanh': nn.Tanh
    }
    assert activation in activations
    return activations[activation]

def build_network(
    inp_dim:int,
    out_dim:int,
    descr=[ #  size layer  activation   initialization
               (16, 'LSTM', 'identity',      None        ),
               (32,   'FC',     'relu', 'kaiming_uniform')]
):
    assert len(descr) > 0
    descr = descr + \
            [(out_dim, 'FC', 'identity', 'kaiming_uniform')]
    layers = []
    for lay_descr in descr:
        out_dim = lay_descr[0]
        lay_fun = lay_descr[1]
        lay_act = lay_descr[2]
        lay_ini = lay_descr[3]
        lay_fn = get_layer(lay_fun, lay_ini)
        act_fn = get_activation(lay_act)
        layers += [lay_fn(inp_dim, out_dim), act_fn()]
        inp_dim = out_dim
    return ( # Return network and recurrent layers
        nn.Sequential(*layers),
        [l for l in layers if isinstance(l,StatefulRNN)]
    )

def build_recurrent_network(
        sizes,
        activation='identity',
        output_activation='identity',
        weight_initialization='kaiming_uniform',
        layer='GRU',
        n_recurrent=1
):
    layer = convert_str_to_torch_layer(layer)
    activation = convert_str_to_torch_functional(activation)
    output_activation = convert_str_to_torch_functional(output_activation)
    layers = list()
    layers_rnn = []
    if n_recurrent == -1:
        n_recurrent = len(sizes) - 2
    for j in range(len(sizes) - 1):
        if j < n_recurrent:
            # Recurrent layer
            lay = layer(sizes[j], sizes[j + 1], batch_first=True)
            lay = StatefulRNN(lay)
            layers_rnn.append(lay)
            act = convert_str_to_torch_functional('identity')
        elif j < len(sizes) - 2:
            # Hidden linear layer
            lay = nn.Linear(sizes[j], sizes[j + 1])
            initialize_layer(weight_initialization, lay)
            act = activation
        else:
            # Output linear layer
            lay = nn.Linear(sizes[j], sizes[j + 1])
            initialize_layer(weight_initialization, lay)
            act = output_activation
        layers += [lay, act()]
    return nn.Sequential(*layers), layers_rnn 

def build_forward_network(
        sizes,
        activation,
        output_activation='identity',
        weight_initialization='kaiming_uniform'
):
    activation = convert_str_to_torch_functional(activation)
    output_activation = convert_str_to_torch_functional(output_activation)
    layers = list()
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        affine_layer = nn.Linear(sizes[j], sizes[j + 1])
        initialize_layer(weight_initialization, affine_layer)
        layers += [affine_layer, act()]
    return nn.Sequential(*layers) 