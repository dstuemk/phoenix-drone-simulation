import argparse
import os
import json
import pathlib
import shutil

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from phoenix_drone_simulation.convert import convert

# Global variables
parent_dir = os.path.realpath(os.path.dirname(__file__))
imgtypes = ['svg', 'png']


def column_name_mapper(name:str):
    return {
        'EpRet/Mean': 'mean(return)',
        'domain_randomization': 'DR',
        'observation_history_size': 'H',
        'observation_model': 'obs_model'
    }.get(name, name)

def load_dataframe(filename,names=None,folder_name=""):
    files_list = list((pathlib.Path(parent_dir) / folder_name).rglob(filename))
    conf_files_list = [ pathlib.Path(p).parent / 'config.json' for p in  files_list]
    df_buffer = []
    for file,conf_file in zip(files_list,conf_files_list):
        df_combined = pd.read_csv(file, delimiter=',', names=names)
        with open(conf_file) as handle:
            dict_conf = json.loads(handle.read())
        # Meta info
        df_combined['filename'] = str(file)
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
    df = load_dataframe('progress.csv',folder_name='log-dir')
    cn = column_name_mapper

    # Training rewards
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

    # Evaluation returns
    df = load_dataframe('returns.csv',['return'],folder_name='log-dir')
    df['run'] = df.index
    grouped = df.groupby([
        cn('filename'),
        cn("observation_history_size"),
        cn("domain_randomization"),
        cn("observation_model"),
        cn("actor")
    ])['return'].mean()
    grouped = grouped.reset_index()
    grouped = grouped.sort_values('return')
    grouped = grouped.drop_duplicates([
        cn("observation_history_size"),
        cn("domain_randomization"),
        cn("observation_model"),
        cn("actor")
    ],keep='last')
    df = df[df.filename.isin(grouped.filename)]
    sns.set_style("darkgrid")
    g = sns.catplot(
        data=df, 
        x=cn("observation_history_size"), y=cn("return"), col=cn("domain_randomization"),
        row=cn("observation_model"), 
        hue=cn("actor"), kind="box",
        height=3, aspect=3/4
    )
    g.set(yscale='symlog')
    g.set(yticks=[-5,-10,-25,-50,-100,-500])
    g.set(yticklabels=[f"{v:.1e}" for v in [-5,-10,-25,-50,-100,-500]])
    g.set(ylim=(df['return'].min() - 50, df['return'].max() + 1))
    for ax in g.axes.ravel():
        ax.set_title(ax.get_title().replace("|", "\n"))
    plt.subplots_adjust(hspace=0.3)
    save_figure("return_evaluation")

    # Automatically export best policies
    folder_dest = pathlib.Path(parent_dir) / "best"
    shutil.rmtree(str(folder_dest), ignore_errors=True)
    for filename in grouped['filename']:
        dirname = pathlib.Path(filename).parent
        print(f"export: {dirname}")
        convert(dirname, 'dat')
        folder_to_copy = pathlib.Path(dirname).parent
        shutil.copytree(
            str(folder_to_copy), 
            str(folder_dest / os.path.basename(folder_to_copy)),
            dirs_exist_ok=True )
    