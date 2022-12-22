""" Implementation of actor classes for RL algorithms.

Author:     Sven Gronauer (sven.gronauer@tum.de), Daniel Stümke (daniel.stuemke@gmail.com)
based on:   Spinning Up's Vanilla Policy Gradient
            https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/vpg/core.py
"""
import numpy as np
import torch
import torch.nn as nn

from torch.distributions.normal import Normal
from phoenix_drone_simulation.algs.net import build_forward_network, build_recurrent_network, build_network

registered_actors = dict()  # global dict that holds pointers to functions 

def register_actor(actor_name):
    """ register actor into global dict"""
    def wrapper(func):
        registered_actors[actor_name] = func
        return func
    return wrapper

def get_registered_actor_fn(actor_type: str, distribution_type: str):
    assert distribution_type == 'categorical' or distribution_type == 'gaussian'
    actor_fn = actor_type + '_' + distribution_type
    msg = f'Did not find: {actor_fn} in registered actors.'
    assert actor_fn in registered_actors, msg
    return registered_actors[actor_fn]

# ====================================
#       Actor Modules
# ====================================

@register_actor("nn_gaussian")
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, layers, final_std=0.01):
        super(Actor, self).__init__()
        log_std = np.log(0.5) * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std),
                                          requires_grad=False)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.final_std = final_std
        self.net, self.layers_rnn = build_network(obs_dim, act_dim, layers)
            
    def reset_states(self):
        for lay_rnn in self.layers_rnn:
            lay_rnn.reset_states()
    
    @property
    def std(self):
        """ Standard deviation of distribution."""
        return torch.exp(self.log_std)    

    def set_log_std(self, frac):
        """ To support annealing exploration noise.
            frac is annealing from 1. to 0 over course of training"""
        assert 0 <= frac <= 1
        new_stddev = (0.5 - self.final_std) * frac + self.final_std # annealing from 0.5 to 0.01
        log_std = np.log(new_stddev) * np.ones(self.act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std),
                                          requires_grad=False)

    def dist(self, obs)-> torch.distributions.Distribution:
        assert self.net is not None
        mu = self.net(obs)
        mu = torch.nan_to_num(mu)
        return Normal(mu, self.std)

    def log_prob_from_dist(self, pi, act) -> torch.Tensor:
        # Last axis sum needed for Torch Normal distribution
        return pi.log_prob(act).sum(axis=-1)

    def forward(self, obs, act=None) -> tuple:
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self.dist(obs)
        logp_a = None
        if act is not None:
            act = torch.nan_to_num(act)
            logp_a = self.log_prob_from_dist(pi, act)
        return pi, logp_a

    def sample(self, obs) -> tuple:
        pi = self.dist(obs)
        a = pi.sample()
        logp_a = self.log_prob_from_dist(pi, a)

        return a, logp_a

    def predict(self, obs) -> tuple:
        """ Predict action based on observation without exploration noise.
            Use this method for evaluation purposes. """
        action = self.net(obs)
        log_p = torch.ones_like(action)  # avoid type conflicts at evaluation

        return action, log_p

    @property
    def is_recurrent(self):
        raise NotImplementedError
