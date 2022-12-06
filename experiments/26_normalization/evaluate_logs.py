import os
import json
import pathlib
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from phoenix_drone_simulation.utils import utils

# Global variables
parent_dir = os.path.realpath(os.path.dirname(__file__))
imgtypes = ['svg', 'png']

def save_figure(name:str):
    for ftype in imgtypes:
        plt.savefig(pathlib.Path(parent_dir) / f"{name}.{ftype}", bbox_inches='tight')

def column_name_mapper(name:str):
    return {
        'EpRet/Mean': 'mean(return)',
        'domain_randomization': 'DR',
        'observation_history_size': 'H',
        'observation_model': 'obs_model',
        'episode_length': 'episode length',
        'use_standardized_obs': 'freeze after [%]'
    }.get(name, name)

def play(ac,env):
    ac.eval()
    done = False
    x = env.reset()
    ret = 0.
    costs = 0.
    episode_length = 0

    while not done:
        obs = torch.as_tensor(x, dtype=torch.float32)
        action, *_ = ac(obs)
        x, r, done, info = env.step(action)
        costs += info.get('cost', 0.)
        ret += r
        episode_length += 1
            
    print(f"Episode \t Return: {ret}\t" + \
        f"Length: {episode_length}\t Costs:{costs}")
    
    return ret,episode_length

if __name__ == '__main__':
    files_list = list((pathlib.Path(parent_dir) / "log-dir").rglob("config.json"))
    df_all = []
    for file_name in files_list:
        with open(file_name) as handle:
            dict_conf = json.loads(handle.read())
        df = pd.DataFrame()
        ckpt = str(file_name.parent)
        ac,env = utils.load_actor_critic_and_env_from_disk(ckpt, model_ckpt='last')
        ep_lens = []
        for run in range(100):
            ret,ep_len = play(ac,env)
            ep_lens.append(ep_len)
        df['episode_length'] = ep_lens
        df['filename'] = str(file_name)
        for k,v in dict_conf.items():
            if not isinstance(v,dict) and not isinstance(v,list):
                df[k] = v
        df_all.append(df)
    df = pd.concat(df_all)
    df.to_pickle(pathlib.Path(parent_dir) / "df.pkl")  

    plt.figure()
    cn = column_name_mapper
    grouped = df.copy()
    grouped = grouped.rename(columns=column_name_mapper)
    grouped[cn("use_standardized_obs")] *= 100
    grouped["actor"] = [v + f" (H={w})" for v,w in zip(df.actor,df.observation_history_size)]
    g = sns.FacetGrid(grouped,col="actor", hue=cn("use_standardized_obs"),col_wrap=2)
    g.map_dataframe(sns.kdeplot, x=cn("episode_length"),fill="True")
    for ax in g.axes.ravel():
        ax.set_xlabel('Episode length')
        ax.set_ylabel('Density')
    g.add_legend()
    save_figure("distribution")