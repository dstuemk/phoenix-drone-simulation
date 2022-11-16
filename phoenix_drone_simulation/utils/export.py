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

#########################################################################
#                                                                       #
#         Convert to keras h5 format                                    #
#                                                                       #
#########################################################################

def convert_kernel_gru(kernel):
    kernel_z, kernel_r, kernel_h = np.vsplit(kernel, 3)
    return np.concatenate((kernel_r.T, kernel_z.T, kernel_h.T), axis=1)

def convert_bias_gru(bias):
    bias = bias.reshape(2, 3, -1) 
    return bias[:, [1, 0, 2], :].reshape((2, -1))

def copy_GRU(lay_tf, lay_torch):
    lay_torch = list(lay_torch.parameters())
    kernel_input = convert_kernel_gru(lay_torch[0].detach())
    kernel_h = convert_kernel_gru(lay_torch[1].detach())
    bias = convert_bias_gru(np.stack((
        lay_torch[2].detach(), 
        lay_torch[3].detach()), axis=0))
    lay_tf.set_weights([
        kernel_input,
        kernel_h,
        bias
    ])

def copy_LSTM(lay_tf, lay_torch):
    lay_torch = list(lay_torch.parameters())
    lay_tf.set_weights([
        lay_torch[0].detach().numpy().transpose(),
        lay_torch[1].detach().numpy().transpose(),
        (lay_torch[2] + lay_torch[3]).detach().numpy()
    ])

def copy_Dense(lay_tf, lay_torch):
    lay_torch = list(lay_torch.parameters())
    lay_tf.set_weights([
        lay_torch[0].detach().numpy().transpose(),
        lay_torch[1].detach().numpy()
    ])

def convert_actor_critic_to_h5(actor_critic: torch.nn.Module,
                               file_path: str,
                               file_name: str = 'model.h5',
                               save_file_path = None,
                               ):
    import tensorflow as tf
    net_torch = actor_critic.pi.net
    model_tf = None
    get_layer = lambda tensor: model_tf.get_layer(
        tensor.name.split('/')[0])
    net_tf = []
    init_fns = []
    # Input Layer
    net_tf.append( tf.keras.layers.Input(
        shape=(1,actor_critic.obs_oms.shape[0]),
        batch_size=1, name='input') )
    # Normalization Layer
    net_tf.append( tf.keras.layers.BatchNormalization(
        center=False, scale=False, epsilon=1e-10, name='batch_normalization')(net_tf[-1]) )
    init_fns.append( lambda lay=net_tf[-1]: get_layer(lay).set_weights([
        actor_critic.obs_oms.mean.numpy(),
        actor_critic.obs_oms.std.numpy()**2
    ]))
    # Hidden Layers
    hidden_ctr = 1
    for module in list(net_torch.modules()):
        # Cells
        if isinstance(module, nn.GRU):
            net_tf.append( tf.keras.layers.GRU(
                units=module.hidden_size, stateful=True, return_sequences=True,
                name=f"layer{hidden_ctr}")(net_tf[-1]) )
            init_fns.append( lambda lay_tf=net_tf[-1],module=module: copy_GRU(
                get_layer(lay_tf), module) )
        elif isinstance(module, nn.LSTM):
            net_tf.append( tf.keras.layers.LSTM(
                units=module.hidden_size, stateful=True, return_sequences=True,
                name=f"layer{hidden_ctr}")(net_tf[-1]) )
            init_fns.append( lambda lay_tf=net_tf[-1],module=module: copy_LSTM(
                get_layer(lay_tf), module) )
        elif isinstance(module, nn.Linear):
            net_tf.append( tf.keras.layers.Dense(units=module.out_features,
                name=f"layer{hidden_ctr}")(net_tf[-1]) )
            init_fns.append( lambda lay_tf=net_tf[-1],module=module: copy_Dense(
                get_layer(lay_tf), module) )
        # Activations
        if isinstance(module, nn.Identity):
            pass
        elif isinstance(module, nn.Tanh):
            net_tf.append(tf.keras.layers.Activation('tanh')(net_tf[-1]))
        elif isinstance(module, nn.ReLU):
            net_tf.append(tf.keras.layers.Activation('relu')(net_tf[-1]))
        hidden_ctr += 1
    # Build network
    model_tf = tf.keras.Model(net_tf[0],net_tf[-1])
    # Set weights
    for init_fn in init_fns:
        init_fn()

    # Check if networks behave in the same way
    actor_critic.training = False
    err = []
    for _ in range(64):
        x = np.random.rand(*actor_critic.obs_shape).astype("float32")
        a_torch,_,_ = actor_critic.step(torch.from_numpy(x))
        a_tf = model_tf(x.reshape((1,1,-1)), training=False)
        err.append((a_torch - a_tf.numpy())**2)
    mse = np.mean(err)
    print(f"Test squared error (avg / max): {mse} / {np.max(err)}")

    if save_file_path is not None:
        file_path = save_file_path
    os.makedirs(file_path, exist_ok=True)
    save_file_name_path = os.path.join(file_path, file_name)
    model_tf.save(save_file_name_path, save_format='h5')
    print(f"Saved model to: {save_file_name_path}")


def convert_actor_critic_to_dat(actor_critic: torch.nn.Module,
                               file_path: str,
                               file_name: str = 'model.dat',
                               save_file_path = None,
                               param_dtype = np.float16
                               ):
    layer_ids = {
        'NORMALIZATION': 0,
        'DENSE'        : 1,
        'LSTM'         : 2,
        'ACTIVATION'   : 3
    }
    activation_ids = {
        'LINEAR'      : 0,
        'RELU'        : 1,
    }
    net_torch = actor_critic.pi.net
    net_arr = []
    layer_arr = []
    # Add normalization layer
    n_in = actor_critic.obs_oms.shape[0]
    n_out = actor_critic.obs_oms.shape[0]
    layer_arr += np.array([ 
        4*actor_critic.obs_oms.shape[0], # N.O. parameters
        layer_ids['NORMALIZATION'],      # Layer type
        n_in,                            # Input size
        n_out                            # Output size 
    ], dtype=np.int32).tobytes()
    layer_arr += actor_critic.obs_oms.mean.numpy().astype(param_dtype).tobytes()         # Mean
    layer_arr += (actor_critic.obs_oms.std.numpy() \
        + actor_critic.obs_oms.eps).astype(param_dtype).tobytes()                        # Std
    layer_arr += np.ones(n_out).astype(param_dtype).tobytes()                            # Gamma
    layer_arr += np.zeros(n_out).astype(param_dtype).tobytes()                           # Beta
    net_arr.append(layer_arr)

    for module in list(net_torch.modules()):
        layer_arr = []
        # Cells
        if isinstance(module, nn.LSTM):
            n_in = n_out
            n_out = module.hidden_size
            layer_arr += np.array([ 
                4*(n_out*n_out + n_in*n_out + n_out) + \
                    2*n_out,                          # N.O. parameters + states
                layer_ids['LSTM'],                    # Layer type
                n_in,                                 # Input size
                n_out                                 # Output size 
            ], dtype=np.int32).tobytes()
            lay_torch = list(module.parameters())
            #           Kernel        Rec. Kernel   Bias
            for mat in [lay_torch[0], lay_torch[1], lay_torch[2] + lay_torch[3]]:
                mat_sel = mat.detach().numpy().astype(param_dtype)
                for mat_idx in range(4):
                    row_range = slice(mat_idx*n_out, (mat_idx+1)*n_out)
                    layer_arr += mat_sel[row_range].transpose().flatten().tobytes()     
            layer_arr += np.array([0.0]*(2*n_out),dtype=param_dtype).tobytes() # States    
            net_arr.append(layer_arr)
        elif isinstance(module, nn.Linear):
            n_in = n_out
            n_out = module.out_features
            layer_arr += np.array([ 
                n_out*(n_in + 1),                     # N.O. parameters
                layer_ids['DENSE'],                   # Layer type
                n_in,                                 # Input size
                n_out                                 # Output size 
            ], dtype=np.int32).tobytes()
            lay_torch = list(module.parameters())
            layer_arr += np.concatenate([
                lay_torch[0].detach().numpy().transpose().flatten(),
                lay_torch[1].detach().numpy().transpose().flatten()
            ], dtype=param_dtype).tobytes()
            net_arr.append(layer_arr)
        # Activations
        if isinstance(module, nn.Identity):
            n_in = n_out
            layer_arr += np.array([ 
                1,                         # N.O. parameters
                layer_ids['ACTIVATION'],   # Layer type
                n_in,                      # Input size
                n_out                      # Output size 
            ], dtype=np.int32).tobytes()
            layer_arr += np.array([
                activation_ids['LINEAR']
            ], dtype=param_dtype).tobytes()
            # Omit linear activation, they are just overhead ...
            # net_arr.append(layer_arr) 
        elif isinstance(module, nn.ReLU):
            n_in = n_out
            layer_arr += np.array([ 
                1,                         # N.O. parameters
                layer_ids['ACTIVATION'],   # Layer type
                n_in,                      # Input size
                n_out                      # Output size 
            ], dtype=np.int32).tobytes()
            layer_arr += np.array([
                activation_ids['RELU']
            ], dtype=param_dtype).tobytes()
            net_arr.append(layer_arr)
    
    net_str = ""
    for line in net_arr:
        net_str += " ".join(str(v) for v in line) + "\n"
    if save_file_path is not None:
        file_path = save_file_path
    os.makedirs(file_path, exist_ok=True)
    save_file_name_path = os.path.join(file_path, file_name)
    print(f"Save model to: {save_file_name_path}")
    print(net_str,  file=open(save_file_name_path, 'w'))

    actor_critic.reset_states()
    inp = [0.0]*actor_critic.obs_oms.shape[0]
    out,_,_ = actor_critic.step(torch.from_numpy(np.array(inp,dtype=np.float32)))
    

