import argparse
import os
import json
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Global variables
parent_dir = os.path.realpath(os.path.dirname(__file__))
imgtypes = ['svg', 'png']


def column_name_mapper(name:str):
    return {
        'EpRet/Mean': 'mean(R)',
        'domain_randomization': 'DR',
        'observation_history_size': 'H',
        'observation_model': 'obs_model'
    }.get(name, name)

def load_dataframe():
    progess_files_list = list(pathlib.Path(parent_dir).rglob('progress.csv'))
    conf_files_list    = list(pathlib.Path(parent_dir).rglob('config.json'))
    df_buffer = []
    for progress_file,conf_file in zip(progess_files_list,conf_files_list):
        df_combined = pd.read_csv(progress_file)
        with open(conf_file) as handle:
            dict_conf = json.loads(handle.read())
        # Copy shallow entries
        for k,v in dict_conf.items():
            if not isinstance(v,dict) and not isinstance(v,list):
                df_combined[k] = v
        # Copy some deeper entries
        df_combined['hidden_sizes'] = "-".join(
            str(v) for v in dict_conf['ac_kwargs']['pi']['hidden_sizes'])
        # Rename architectures
        df_combined.actor = [
            {
                'forward': 'FNN', 
                'recurrent': 'RNN'
            }.get(v,v) for v in df_combined.actor]
        df_buffer.append(df_combined)
    return pd.concat(df_buffer).rename(columns=column_name_mapper)

def save_figure(name:str):
    for ftype in imgtypes:
        plt.savefig(pathlib.Path(parent_dir) / f"{name}.{ftype}", bbox_inches='tight')

if __name__ == '__main__':
    df = load_dataframe()
    cn = column_name_mapper

    plt.figure()
    g = sns.relplot(
        data=df[df.Epoch % 5 == 0], 
        x=cn("Epoch"), y=cn("EpRet/Mean"), col=cn("domain_randomization"),
        row=cn("observation_model"), hue=cn("actor"), 
        style=cn("observation_history_size"), kind="line",
        height=3, aspect=3/4
    )
    for ax in g.axes.ravel():
        ax.set_title(ax.get_title().replace("|", "\n"))
    plt.subplots_adjust(hspace=0.3)
    save_figure("reward_training")



