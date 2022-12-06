import os
import json
import pathlib
import torch
import pandas as pd
from phoenix_drone_simulation.utils import utils

# Global variables
parent_dir = os.path.realpath(os.path.dirname(__file__))

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
        for dr_target in [0.0, 0.1, 0.2]:
            with open(file_name) as handle:
                dict_conf = json.loads(handle.read())
            df = pd.DataFrame()
            ckpt = str(file_name.parent)
            ac,env = utils.load_actor_critic_and_env_from_disk(ckpt, model_ckpt='best')
            dr_source = env.env.domain_randomization
            env.env.domain_randomization = dr_target
            ep_lens = []
            ep_rets  = []
            for run in range(100):
                ret,ep_len = play(ac,env)
                ep_lens.append(ep_len)
                ep_rets.append(ret)
            df['episode_length'] = ep_lens
            df['episode_return'] = ep_rets
            df['dr_source'] = dr_source
            df['dr_target'] = dr_target
            df['filename'] = str(file_name)
            for k,v in dict_conf.items():
                if not isinstance(v,dict) and not isinstance(v,list):
                    df[k] = v
            df_all.append(df)
    df = pd.concat(df_all)
    df.to_pickle(pathlib.Path(parent_dir) / "df.pkl")  

    grouped = df.groupby(["actor", "observation_history_size", "dr_source", "dr_target", "observation_model"])
    grouped = grouped.agg({"episode_length": ["mean", "median"], "episode_return": ["mean", "median"]}).reset_index()
    pd.pivot_table(grouped, values=["episode_return", "episode_length"], 
        index=['observation_model','actor','observation_history_size'], 
        columns=['dr_source','dr_target']).to_excel(pathlib.Path(parent_dir) / "table.xlsx")