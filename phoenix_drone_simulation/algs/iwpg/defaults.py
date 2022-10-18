"""
    Define default parameters for Importance-weighted Policy Gradient (IWPG)
    algorithm.
"""


def defaults():
    return dict(
        actor='recurrent',
        critic='recurrent',
        ac_kwargs={
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
        },
        adv_estimation_method='gae',
        epochs=300,
        gamma=0.99,
        steps_per_epoch=32 * 1000,
        # Early stopping criterion adds robustness towards hyper-parameters
        # see "Successful ingredients" Paper
        use_kl_early_stopping=True,
    )


def locomotion():
    """Default hyper-parameters for Bullet's locomotion environments."""
    params = defaults()
    params['epochs'] = 312
    params['max_ep_len'] = 1000
    params['pi_lr'] = 3e-4  # default choice is Adam
    params['steps_per_epoch'] = 8 * 1000
    params['vf_lr'] = 3e-4  # default choice is Adam
    return params


# Hack to circumvent kwarg errors with the official PyBullet Envs
def gym_locomotion_envs():
    params = locomotion()
    return params


def gym_manipulator_envs():
    """Default hyper-parameters for Bullet's manipulation environments."""
    params = defaults()
    params['epochs'] = 312
    params['max_ep_len'] = 150
    params['pi_lr'] = 3e-4  # default choice is Adam
    params['steps_per_epoch'] = 32 * 1000
    params['vf_lr'] = 3e-4  # default choice is Adam
    return params
