import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from phoenix_drone_simulation.utils import utils

# Global variables
parent_dir = os.path.realpath(os.path.dirname(__file__))

def play(ac,env):
    ac.eval()

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            if isinstance(model,nn.LSTM):
                activation[name] = torch.concat(output[1]).view(-1).detach()
            else:
                activation[name] = output.detach()
        return hook
    
    lay = [m for m in ac.pi.modules() if (
        isinstance(m,nn.Linear) or isinstance(m,nn.LSTM))][0]
    lay.register_forward_hook(get_activation('act1'))

    n_episodes = 100
    X = deque(maxlen=n_episodes)
    M = deque(maxlen=n_episodes)
    Y = deque(maxlen=n_episodes)

    for i in range(n_episodes):
        done = False
        x = env.reset()
        ret = 0.
        costs = 0.
        episode_length = 0
        X.append([])
        M.append([])
        Y.append([])
        while not done:
            obs = torch.as_tensor(x, dtype=torch.float32)
            action, *_ = ac(obs)
            X[-1].append(x)
            Y[-1].append(np.concatenate([env.drone.rpy, env.drone.xyz_dot, env.drone.xyz[0:2]]))
            x, r, done, info = env.step(action)
            M[-1].append(activation['act1'].numpy())
            costs += info.get('cost', 0.)
            ret += r
            episode_length += 1
            
        print(f"Episode {i+1}\t Return: {ret}\t" + \
            f"Length: {episode_length}\t Costs:{costs}")
    
    X = np.vstack(X)
    Y = np.vstack(Y)
    M = np.vstack(M)
    return X,Y,M

def collect_data(ckpt_fnn,ckpt_rnn):
    for id,ckpt in zip(['fnn', 'rnn'],[ckpt_fnn, ckpt_rnn]):  
        ac, env = utils.load_actor_critic_and_env_from_disk(ckpt)
        X,Y,M = play(ac,env)
        # Extend with current observation
        M = np.hstack([M,X[:,0:X.shape[1] // env.observation_history_size]])
        np.save(os.path.join(parent_dir, f"X-{id}.npy"), X)
        np.save(os.path.join(parent_dir, f"Y-{id}.npy"), Y)
        np.save(os.path.join(parent_dir, f"M-{id}.npy"), M)
        # Generate baseline dataset
        M_base = X
        for H in [1,2,4,8]:
            for h in range(H):
                M_base = np.hstack([X, np.roll(M_base, shift=1, axis=0)])
            np.save(os.path.join(parent_dir, f"X-base{H}.npy"), X)
            np.save(os.path.join(parent_dir, f"Y-base{H}.npy"), Y)
            np.save(os.path.join(parent_dir, f"M-base{H}.npy"), M_base)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--ckpt-rnn', type=str, required=True,
                        help='Name path of the rnn agent.}')
    parser.add_argument('--ckpt-fnn', type=str, required=True,
                        help='Name path of the fnn agent.}')
    args = parser.parse_args()

    collect_data(args.ckpt_fnn, args.ckpt_rnn)
    
    print("FIN")