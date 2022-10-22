
import phoenix_drone_simulation
from phoenix_drone_simulation.train import get_training_command_line_args
from phoenix_drone_simulation.benchmark import Benchmark

ALG = 'ppo'
ENV = 'DroneCircleBulletEnv-v0'
NUM_RUNS = 5

env_specific_kwargs = {
    ENV: {
        'epochs': 250,
        'steps_per_epoch': 32000,
        'latency':	0.02,              # From Zero-shot paper
        'observation_noise': 1,        # sensor noise enabled when > 0
        'motor_time_constant': 0.120,  # [s]
        'motor_thrust_noise': 0.05,    # noise in % added to thrusts
        'penalty_spin': 0.001,
        'max_ep_len': 200,
        'seq_len': 100,
        'seq_overlap': 50,
        'save_freq': 25,
        'observe_position': False,
        'observation_history_size': 1, 
        'use_standardized_obs': 1.0,
    },
}

common_grid_dict = {
    'domain_randomization': [0.1, 0.0],
    'observation_model': ['sensor', 'state'],
}

# ------------------------------------------------------------------------------
#   Recurrent architecture
# ------------------------------------------------------------------------------

recurrent_grid_dict = {
    **common_grid_dict,  
    'actor': ['recurrent'],
    'critic': ['recurrent'],
    'ac_kwargs': [
        {
            'pi': {
                'activation': 'identity', 
                'hidden_sizes': [20, 20],
                'layer': 'GRU'
            }, 
            'val': {
                'activation': 'identity', 
                'hidden_sizes': [128, 128],
                'layer': 'GRU'
            }
        }
    ]
}

# ------------------------------------------------------------------------------
#   Forward architecture
# ------------------------------------------------------------------------------

forward_grid_dict = {
    **common_grid_dict,
    'observation_history_size': [2, 4, 8],
    'actor': ['forward'],
    'critic': ['forward'],
    'ac_kwargs': [
        {
            'pi': {
                'activation': 'relu', 
                'hidden_sizes': [32, 32]
            }, 
            'val': {
                'activation': 'tanh', 
                'hidden_sizes': [300, 300]
            }
        },
    ]
}

# ------------------------------------------------------------------------------
#   Main program
# ------------------------------------------------------------------------------


def train(args,parameter_grid_dict):
    alg_setup = {
        ALG: parameter_grid_dict,
    }
    bench = Benchmark(
        alg_setup,
        env_ids=list(env_specific_kwargs.keys()),
        log_dir=args.log_dir,
        num_cores=args.cores,
        num_runs=NUM_RUNS,
        env_specific_kwargs=env_specific_kwargs,
        use_mpi=True,
        init_seed=0,  # start with seed 0 and then count up
    )
    bench.run()

if __name__ == '__main__':
    args, unparsed_args = get_training_command_line_args(
        alg=ALG, env=ENV)
    train(args,forward_grid_dict)
    train(args,recurrent_grid_dict)