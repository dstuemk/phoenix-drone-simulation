"""Run RL training with Proximal Policy Optimization (PPO) using multi-cores.

Author:     Sven Gronauer (sven.gronauer@tum.de)
Created:    16.11.2021
"""
import argparse
import numpy as np
import psutil
import sys
import gym
import getpass
import time
import torch

# local imports:
import phoenix_drone_simulation  # necessary to load our custom drone environments
from phoenix_drone_simulation.algs.model import Model
from phoenix_drone_simulation.utils.mpi_tools import mpi_fork


def main():
    USE_CORES = 4

    # Let us count the number of physical cores on this machine...
    # Exclude hyper-threading and round cores to anything in: [2, 4, 8, 16, ...]
    physical_cores = 2 ** int(np.log2(psutil.cpu_count(logical=False)))

    # Use number of physical cores as default. If also hardware threading CPUs
    # should be used, enable this by the use_number_of_threads=True
    use_number_of_threads = True if USE_CORES > physical_cores else False
    if mpi_fork(USE_CORES, use_number_of_threads=use_number_of_threads):
        # Re-launches the current script with workers linked by MPI
        sys.exit()

    env_id = "DroneCircleBulletEnv-v0"
    user_name = getpass.getuser()

    # Create a seed for the random number generator
    random_seed = int(time.time()) % 2 ** 16

    # I usually save my results into the following directory:
    default_log_dir = f"/var/tmp/{user_name}"

    # 1) Setup learning model
    model = Model(
        alg='ppo',  # choose between: ppo
        env_id=env_id,
        log_dir=default_log_dir,
        init_seed=random_seed,
        use_mpi=True,  # set this parameter True to use all available CPU cores
        algorithm_kwargs={
                'steps_per_epoch': 64000,
                'latency':      0.02,
                'observation_noise': 1,
                'motor_time_constant': 0.120,
                'motor_thrust_noise': 0.05,
                'observation_model': 'state',
                'domain_randomization': 0.1,
                'observation_history_size': 1,
                'use_standardized_obs': 1/4,
                'max_ep_len': 500,
                'seq_len': 64,
                'seq_overlap': 32,
        }
    )

    # compile model and define the number of CPU cores to use for training
    model.compile(num_cores=USE_CORES)

    # 2) Train model - it takes typically at least 100 epochs for training
    model.fit(epochs=250)

    # 3) Benchmark the final policy and save results into `returns.csv`
    model.eval()


if __name__ == '__main__':
    main()

