import os
import json
import pathlib
from collect_data import collect_data
from train_decoder import train_decoder

# Global variables
parent_dir = os.path.realpath(os.path.dirname(__file__))

if __name__ == '__main__':
    conf_files_list = list((pathlib.Path(parent_dir) / 'log-dir').rglob('config.json'))
    rnn_ckpts = []
    fnn_ckpts = []
    for conf_file in conf_files_list:
        with open(conf_file) as handle:
            dict_conf = json.loads(handle.read())
        if dict_conf['actor'] == 'forward':
            fnn_ckpts.append(str(conf_file.parent))
        elif dict_conf['actor'] == 'recurrent':
            rnn_ckpts.append(str(conf_file.parent))
        else:
            print(f"Invalid actor: {dict_conf['actor']}")
    first = True
    for fnn_ckpt,rnn_ckpt in zip(fnn_ckpts,rnn_ckpts):
        collect_data(fnn_ckpt, rnn_ckpt)
        train_decoder(append_logs=not first)
        first = False
    
    print("FIN")
        
