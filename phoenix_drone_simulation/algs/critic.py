""" Implementation of critic classes for RL algorithms.

Author:     Sven Gronauer (sven.gronauer@tum.de), Daniel Stümke (daniel.stuemke@gmail.com)
based on:   Spinning Up's Vanilla Policy Gradient
            https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/vpg/core.py
"""
import torch
import torch.nn as nn

from phoenix_drone_simulation.algs.net import build_forward_network, build_recurrent_network, build_network

registered_critics = dict()  # global dict that holds pointers to functions 

def register_critic(critic_name):
    """ register critic into global dict """
    def wrapper(func):
        registered_critics[critic_name] = func
        return func
    return wrapper

def get_registered_critic_fn(critic_type: str):
    critic_fn = critic_type
    msg = f'Did not find: {critic_fn} in registered critics.'
    assert critic_fn in registered_critics, msg
    return registered_critics[critic_fn]

# ====================================
#       Critic Modules
# ====================================

@register_critic("nn")
class Critic(nn.Module):
    def __init__(self, obs_dim, layers):
        super(Critic, self).__init__()
        self.net, self.layers_rnn = build_network(obs_dim, 1, layers)
            
    def reset_states(self):
        for lay_rnn in self.layers_rnn:
            lay_rnn.reset_states()

    def forward(self, obs):
        assert self.net is not None
        return torch.squeeze(self.net(obs),
                             -1)  # Critical to ensure v has right shape.
                             
    @property
    def is_recurrent(self):
        raise NotImplementedError