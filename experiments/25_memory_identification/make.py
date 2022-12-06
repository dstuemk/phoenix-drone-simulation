import os
import json
import pathlib
from collect_data import collect_data
from train_decoder import train_decoder

# Global variables
parent_dir = os.path.realpath(os.path.dirname(__file__))

if __name__ == '__main__':
    conf_files_list = list((pathlib.Path(parent_dir) / 'log-dir').rglob('config.json'))
    dr_ckpts = []
    nodr_ckpts = []
    for conf_file in conf_files_list:
        with open(conf_file) as handle:
            dict_conf = json.loads(handle.read())
        if dict_conf['domain_randomization'] > 0:
            dr_ckpts.append(str(conf_file.parent))
        else:
            nodr_ckpts.append(str(conf_file.parent))
    first = True
    for dr_ckpt,nodr_ckpt in zip(dr_ckpts,nodr_ckpts):
        collect_data(dr_ckpt, nodr_ckpt)
        train_decoder(append_logs=not first)
        first = False
    
    print("FIN")
        
