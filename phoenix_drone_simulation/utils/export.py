r"""Export functionalities for custum CrazyFlie Firmware.

"""
import os
import json
import gym
import numpy as np
import torch
import torch.nn as nn

# local imports
import phoenix_drone_simulation
import phoenix_drone_simulation.utils.loggers as loggers
from phoenix_drone_simulation.algs import core
from phoenix_drone_simulation.utils.utils import get_file_contents


def count_vars(module: nn.Module):
    r"""Count number of variables in Neural Network."""
    return sum([np.prod(p.shape) for p in module.parameters()])

def convert_actor_critic_to_h5(actor_critic: torch.nn.Module,
                               file_path: str,
                               file_name: str = 'model.h5'
                               ):
    net_torch = actor_critic.pi.net
    net_tf    = actor_critic.pi.net_tf

    net_tf.layers[1].set_weights([
        # Normalization layer
        actor_critic.obs_oms.mean.numpy(),
        actor_critic.obs_oms.std.numpy()**2
    ])

    tf_layers = net_tf.layers
    for module in list(net_torch.modules()):
        if isinstance(module, nn.GRU):
            gru_idx = np.where([isinstance(el, tf.keras.layers.GRU) for el in tf_layers])[0][0]
            lay_torch = list(module.parameters())
            kernel_input = convert_kernel_gru(lay_torch[0].detach())
            kernel_h = convert_kernel_gru(lay_torch[1].detach())
            bias = convert_bias_gru(np.stack((
                lay_torch[2].detach(), 
                lay_torch[3].detach()), axis=0))
            lay_tf = tf_layers[gru_idx]
            lay_tf.set_weights([
                kernel_input,
                kernel_h,
                bias
            ])
            del tf_layers[gru_idx]
        elif isinstance(module, nn.LSTM):
            lstm_idx = np.where([isinstance(el, tf.keras.layers.LSTM) for el in tf_layers])[0][0]
            lay_torch = list(module.parameters())
            lay_tf = tf_layers[lstm_idx]
            lay_tf.set_weights([
                lay_torch[0].detach().numpy().transpose(),
                lay_torch[1].detach().numpy().transpose(),
                (lay_torch[2] + lay_torch[3]).detach().numpy()
            ])
            del tf_layers[lstm_idx]
        elif isinstance(module, nn.Linear):
            dense_idx = np.where([isinstance(el, tf.keras.layers.Dense) for el in tf_layers])[0][0]
            lay_torch = list(module.parameters())
            lay_tf = tf_layers[dense_idx]
            lay_tf.set_weights([
                lay_torch[0].detach().numpy().transpose(),
                lay_torch[1].detach().numpy()
            ])
            del tf_layers[dense_idx]

    # Check if networks behave in the same way
    actor_critic.training = False
    err = []
    a_pre = None
    for step_idx in range(64):
        x = np.random.rand(*actor_critic.obs_shape).astype("float32")
        a_torch,_,_ = actor_critic.step(torch.from_numpy(x))
        a_tf = net_tf(x.reshape((1,1,-1)), training=False)
        err.append((a_torch - a_tf.numpy())**2)
    mse = np.mean(err)
    print(f"Test squared error (avg / max): {mse} / {np.max(err)}")

    save_file_name_path = os.path.join(file_path, file_name)
    net_tf.save(save_file_name_path, save_format='h5')
    print(f"Saved model to: {save_file_name_path}")

