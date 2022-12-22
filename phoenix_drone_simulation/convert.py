r"""Convert Torch models into export file formats like JSON.

This is a function used by Sven to extract the policy networks from trained
Actor-Critic modules and convert it to the JSON file format holding NN parameter
values.

Important Note:
    this file assumes that you are using the CrazyFlie Firmware adopted the the
    Chair of Data Processing - Technical University Munich (TUM)
"""

import argparse
import os
from phoenix_drone_simulation.utils import utils
from phoenix_drone_simulation.utils import export

def convert(ckpt,output,save_file_path=None):
    assert os.path.exists(ckpt)
    if output == 'h5':
        ac, env = utils.load_actor_critic_and_env_from_disk(ckpt)
        export.convert_actor_critic_to_h5(
            actor_critic=ac,
            file_path=ckpt,
            save_file_path=save_file_path
        )
    elif output == 'json':
        ac, env = utils.load_actor_critic_and_env_from_disk(ckpt)
        export.convert_actor_critic_to_json(
            actor_critic=ac,
            file_path=ckpt,
            save_file_path=save_file_path
        )
    else:
        raise ValueError('Unexpected file output.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Name path of the file to be converted.}')
    parser.add_argument('--output', type=str, default='json',
                        help='Choose output file format: [h5, json].}')
    args = parser.parse_args()

    convert(args.ckpt,args.output)