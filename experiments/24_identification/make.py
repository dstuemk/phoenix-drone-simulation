import os
import json
import pathlib
import random
import numpy as np
import pandas as pd
from phoenix_drone_simulation.utils import utils

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn import svm,neural_network,gaussian_process

# Global variables
parent_dir = os.path.realpath(os.path.dirname(__file__))

def play(ac,env):
    ac.eval()
    done = False
    x = env.reset()
    ret = 0.
    costs = 0.
    episode_length = 0

    x_buf = []
    m_buf = []

    activation = {}
    def getActivation(name):
        # the hook signature
        def hook(model, input, output):
            activation[name] = torch.concat([v for v in output[1]]).view(-1).detach()
        return hook
    
    for module in list(ac.pi.modules()):
        if isinstance(module, nn.LSTM):
            module.register_forward_hook(getActivation("LSTM"))

    while not done:
        x_buf.append(x)
        obs = torch.as_tensor(x, dtype=torch.float32)
        action, *_ = ac(obs)
        m_buf.append(np.array(activation.get("LSTM", [0])))
        x, r, done, info = env.step(action)
        costs += info.get('cost', 0.)
        ret += r
        episode_length += 1
            
    print(f"Episode \t Return: {ret}\t" + \
        f"Length: {episode_length}\t Costs:{costs}")
    
    return ret,episode_length,np.array(x_buf),np.array(m_buf)

if __name__ == '__main__':
    files_list = list((pathlib.Path(parent_dir) / "log-dir").rglob("config.json"))
    latency_M = []
    latency_X = []
    latency_true = []
    for file_name in files_list:
        X_data = []
        M_data = []
        Y_data = []
        for latency in [0.005, 0.015, 0.025]:
            ckpt = str(file_name.parent)
            ac,env = utils.load_actor_critic_and_env_from_disk(ckpt, model_ckpt='best')
            env.env.domain_randomization = 0
            env.env.drone.set_latency(latency)
            run = 0
            while run < 100:
                ret,ep_len,X,M = play(ac,env)
                if ep_len >= 500:
                    for idx in range(99,500,10):
                        X_data.append(X[idx])
                        M_data.append(M[idx])
                        Y_data.append(latency)
                    run += 1
        X_data = np.vstack(X_data)
        M_data = np.vstack(M_data)
        scaler_X = StandardScaler()
        scaler_X.fit(X_data)
        X_data = scaler_X.transform(X_data)
        scaler_M = StandardScaler()
        scaler_M.fit(M_data)
        M_data = scaler_M.transform(M_data)
        
        Y_data = (np.vstack(Y_data)*1000).astype(int)
        indices = list(range(Y_data.shape[0]))
        n_splits = 10
        split_size = int(np.ceil(len(indices)) / n_splits)
        random.Random(42).shuffle(indices)
        indices = np.array(indices)
        for split_idx in range(n_splits):
            idx_start = split_idx*split_size
            idx_end   = (split_idx+1)*split_size
            train_indices = [
                *list(range(0,idx_start)),
                *list(range(idx_end,len(indices)))
            ]
            test_indices = list(range(idx_start,idx_end))
            train_indices = indices[train_indices]
            test_indices = indices[test_indices]
            clf_M = neural_network.MLPClassifier(hidden_layer_sizes=(500,500,),random_state=0,verbose=True,early_stopping=True)
            clf_X = neural_network.MLPClassifier(hidden_layer_sizes=(500,500,),random_state=0,verbose=True,early_stopping=True)

            clf_M.fit(M_data[train_indices],Y_data[train_indices])
            y_pred_M = clf_M.predict(M_data[test_indices])

            clf_X.fit(X_data[train_indices],Y_data[train_indices])
            y_pred_X = clf_X.predict(X_data[test_indices])
            
            y_true = Y_data[test_indices]

            latency_M = [*latency_M, *y_pred_M.ravel()]
            latency_X = [*latency_X, *y_pred_X.ravel()]
            latency_true = [*latency_true, *y_true.ravel()]
    
    df = pd.DataFrame()
    df['latency_true'] = latency_true
    df['latency_M'] = latency_M
    df['latency_X'] = latency_X

    df.to_pickle(pathlib.Path(parent_dir) / "data.pkl")

    print(f"Accuracy M: { (np.array(latency_M) == np.array(latency_true)).mean() }")
    print(f"Accuracy X: { (np.array(latency_X) == np.array(latency_true)).mean() }")

    print("FIN")