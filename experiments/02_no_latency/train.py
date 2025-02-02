import phoenix_drone_simulation
from phoenix_drone_simulation.train import get_training_command_line_args
from phoenix_drone_simulation.benchmark import Benchmark

ALG = 'ppo'
ENV = 'DroneCircleBulletEnv-v0'
NUM_RUNS = 5

env_specific_kwargs = {
    ENV: {
        'epochs': 500,
        'steps_per_epoch': 64000,
        'latency':      0.00,          # From Zero-shot paper: 0.02
        'observation_noise': 1,        # sensor noise enabled when > 0
        'motor_time_constant': 0.120,  # [s]
        'motor_thrust_noise': 0.05,    # noise in % added to thrusts
        'penalty_spin': 0.001,
        'max_ep_len': 500,
        'seq_len': 64,
        'seq_overlap': 32,
        'save_freq': 25,
        'observe_position': True,
        'observation_history_size': 1,
        'use_standardized_obs': 1/4,
        'randomize_latency': -1,       # Experimental
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
    'ac_kwargs': [
        {
            'pi':  [ #  size layer  activation   initialization
                       (16, 'LSTM', 'identity',      None        ),
                       (32,   'FC',     'relu', 'kaiming_uniform')
            ],
            'val': [ #  size layer  activation   initialization
                      (128, 'LSTM', 'identity',      None        ),
                      (300,   'FC',     'relu', 'kaiming_uniform')
            ]
        }
    ]
}

# ------------------------------------------------------------------------------
#   Forward architecture
# ------------------------------------------------------------------------------

forward_grid_dict = {
    **common_grid_dict,
    'observation_history_size': [2, 4, 8],
    'ac_kwargs': [
        {
            'pi': [ #  size layer  activation   initialization
                          (32,   'FC',     'relu', 'kaiming_uniform'),
                          (32,   'FC',     'relu', 'kaiming_uniform')
            ],
            'val': [ #  size layer  activation   initialization
                          (300,   'FC',     'relu', 'kaiming_uniform'),
                          (300,   'FC',     'relu', 'kaiming_uniform')
            ],
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
    train(args,recurrent_grid_dict)
    train(args,forward_grid_dict)