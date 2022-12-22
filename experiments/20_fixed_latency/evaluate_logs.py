import argparse
import os
import json
import pathlib
import shutil
import itertools

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from phoenix_drone_simulation.utils import utils
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

def load_dataframe(filename,names=None,folder_name="", trafo=lambda x:x):
    files_list = list((pathlib.Path(parent_dir) / folder_name).rglob(filename))
    conf_files_list = [ pathlib.Path(p).parent / 'config.json' for p in  files_list]
    df_buffer = []
    for file,conf_file in zip(files_list,conf_files_list):
        df_combined = trafo(pd.read_csv(file, delimiter=',', names=names))
        df_combined['idx'] = list(range(df_combined.shape[0]))
        with open(conf_file) as handle:
            dict_conf = json.loads(handle.read())
        # Meta info
        df_combined['filename'] = str(file)
        df_combined['conf_filename'] = str(conf_file)
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

def calculate_reward(df_flight:pd.DataFrame,reset_time=True):
    alpha = 2.0*np.pi/3.0
    net_time = df_flight['time'] - np.min(df_flight['time'])*reset_time
    t = net_time.to_numpy()
    t = np.linspace(t[0],t[-1],t.size)
    z_err = 1.0 - df_flight['z']
    y_err = 0.25*np.sin(net_time * alpha) - df_flight['y']
    x_err = 0.25*(1.0 - np.cos(net_time * alpha)) - df_flight['x']
    # Compute reward
    penalties = []
    penalty_action_ = 1e-4
    penalty_spin_ = 1e-3
    penalty_velocity_ = 1e-4
    ARP_ = 1e-3
    last_action = None
    for step in range(min(t.size, 500)):
        action = np.array([
            df_flight.iloc[step]['M1'],
            df_flight.iloc[step]['M2'],
            df_flight.iloc[step]['M3'],
            df_flight.iloc[step]['M4']])
        act_diff = 0 if last_action is None else action - last_action
        last_action = action
        normed_clipped_a = 0.5 * (np.clip(action, -1, 1) + 1)

        rpy_dot = np.array([
            df_flight.iloc[step]['roll_dot'],
            df_flight.iloc[step]['pitch_dot'],
            df_flight.iloc[step]['yaw_dot']])

        xyz_dot = np.array([
            df_flight.iloc[step]['x_dot'],
            df_flight.iloc[step]['y_dot'],
            df_flight.iloc[step]['z_dot']])

        penalty_action = penalty_action_ * np.linalg.norm(normed_clipped_a)
        penalty_action_rate = ARP_ * np.linalg.norm(act_diff)
        penalty_spin = penalty_spin_ * np.linalg.norm(rpy_dot)
        penalty_velocity = penalty_velocity_ * np.linalg.norm(xyz_dot)

        xyz_err = np.array([
            x_err.iloc[step], 
            y_err.iloc[step], 
            z_err.iloc[step]])
        dist = np.linalg.norm(xyz_err)

        penalties.append({
            'action': penalty_action,
            'action_rate': penalty_action_rate,
            'spin': penalty_spin,
            'velocity': penalty_velocity,
            'distance': dist,
            'terminal': 0.0
        })
    
    if t.size < 500: # 5 sec
        penalties[-1]['terminal'] = 100
    
    sum = 0.0
    for penalty in penalties:
        penalty['sum'] = np.sum([v for v in penalty.values()])
        sum += penalty['sum']
    
    #return pd.DataFrame(penalties)
    return pd.DataFrame({'Reward': [-sum]})
    
if __name__ == '__main__':
    cn = column_name_mapper
    
    # PID (baseline)
    df_pid = load_dataframe('pid__*.csv',folder_name='log-dir',trafo=calculate_reward)

    # Real-world rewards
    df = load_dataframe('flight__*.csv',folder_name='log-dir',trafo=calculate_reward)
    grouped = df.copy()
    grouped[cn("actor")] = [f"{pi} (H={h})" for pi,h in zip(
        grouped[cn("actor")], grouped[cn("observation_history_size")]
    )]
    plt.figure()
    sns.set_style("darkgrid")
    g = sns.catplot(
        data=grouped, 
        x=cn("domain_randomization"), y=cn("Reward"), col=cn("observation_model"),
        row=cn("latency"), hue=cn("actor"), kind="box",
        height=3, aspect=1
    )
    g.set(yscale='symlog')
    g.set(yticks=[-5,-10,-25,-50,-100,-200])
    g.set(yticklabels=[f"{v:.0f}" for v in [-5,-10,-25,-50,-100,-200]])
    g.set(ylim=(df['Reward'].min() - 50, df['Reward'].max() + 1))
    pid = df_pid.Reward
    for ax in g.axes.ravel():
        bar_y = float(pid.quantile(0.75))
        bar_h = float(pid.quantile(0.25)) - bar_y
        xlim = ax.get_xlim()
        ax.barh(bar_y,999,bar_h,left=-99,align='edge', color='k', alpha=0.2)
        ax.barh(-100,999,-300,left=-99,align='edge', 
            color='none', edgecolor='k', hatch='x', alpha=0.5)
        ax.axhline(
            float(pid.median()),
            ls="--", color='k')
        ax.set_xlim(xlim)
    save_figure(f"reward_flights")


    # Real-world success rate
    df = load_dataframe('flight__*.csv',folder_name='log-dir')
    grouped = df.sort_values('idx')
    grouped = grouped.drop_duplicates([
        cn("filename")
    ], keep='last')
    grouped['flight time'] = np.clip(grouped['idx'] / 100, 0, 20)
    grouped['success'] = grouped['flight time'] >= 19.95
    grouped = grouped.groupby([
        cn("observation_model"), 
        cn("domain_randomization"),
        cn("actor"),  
        cn("observation_history_size")]).agg({
            "success": [lambda x: np.sum(x) / x.shape[0] * 100],  
            "flight time": ["min", "max", "mean", "median"]
    })
    grouped.to_excel(pathlib.Path(parent_dir) / "success_rate.xlsx")

    # Real-world trajectories
    sns.set_style("whitegrid")
    df = load_dataframe('flight__*.csv',folder_name='log-dir')
    t = np.linspace(0,np.pi*2,300)
    z = np.array([1]*t.size)
    y = 0.25*np.sin(t)
    x = 0.25*(1.0 - np.cos(t))
    ctr = 0
    for fn in df[cn("filename")].unique():
        df_traj = df[df[cn("filename")] == fn]
        df_traj = df_traj.sort_values(cn("time"))
        if df_traj.shape[0] < 500: 
            continue
        df_traj = df_traj.iloc[200:500]
        normalized_thrust = np.clip(np.array([
                df_traj.M1, df_traj.M2, df_traj.M3, df_traj.M4
            ]) + 1.0, 0, 2)
        col = np.linalg.norm(normalized_thrust,axis=0)/4
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(x,y,z, 'red')
        for i in range(df_traj.shape[0]-1):
            ax.plot(df_traj.x[i:i+2], df_traj.y[i:i+2], df_traj.z[i:i+2], 
            color=plt.cm.winter(col[i]))
        ax.set_zlim(0.9,1.1)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.winter)
        cbar = plt.colorbar(sm, pad=0.1, fraction=0.015)
        cbar.ax.set_ylabel('thrust %', labelpad=5)
        plt.xlabel("x")
        plt.ylabel("y")
        ax.set_zlabel("z")
        plt.title(f"DR = {df_traj.iloc[0][cn('domain_randomization')]} | " + \
            df_traj.iloc[0][cn('observation_model')] + " | " + \
            df_traj.iloc[0][cn('actor')] + \
            f" (H={df_traj.iloc[0][cn('observation_history_size')]})")
        save_figure(f"traj_{ctr:03d}" + df_traj.iloc[0][cn('actor')] + "_H" + \
            str(df_traj.iloc[0][cn('observation_history_size')]) + \
            "DR_" + str(int(df_traj.iloc[0][cn('domain_randomization')]*10)) + \
            df_traj.iloc[0][cn('observation_model')])
        ctr += 1

    # Training rewards
    df = load_dataframe('progress.csv',folder_name='log-dir')
    plt.figure()
    sns.set_style("darkgrid")
    g = sns.relplot(
        data=df[df.Epoch % 5 == 0], 
        x=cn("Epoch"), y=cn("EpRet/Mean"), col=cn("domain_randomization"),
        row=cn("observation_model"), hue=cn("actor"), 
        style=cn("observation_history_size"), kind="line",
        height=3, aspect=1
    )
    for ax in g.axes.ravel():
        ax.set_title(ax.get_title().replace("|", "\n"))
    plt.subplots_adjust(hspace=0.3)
    save_figure("reward_training")
    