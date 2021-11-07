import sys,os
import numpy as np
import torch
import random
from lib.training.training import read_config_from_file
from lib.training.importer import import_scheme

def run_worker(rank, world_size, command, SCHEME, config, seed):
    torch.cuda.set_device(rank)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.distributed.init_process_group(backend="nccl", 
                                         rank=rank,
                                         world_size=world_size)
    
    print(f'Initiated rank: {rank}', flush=True)
    try:
        scheme = SCHEME(config, rank, world_size)
        
        if command == 'train':
            scheme.execute_training()
        elif command == 'predict':
            scheme.make_predictions()
        elif command == 'evaluate':
            scheme.do_evaluations()
    finally:
        torch.distributed.destroy_process_group()
        print(f'Rank {rank}:Destroyed process!', flush=True)
    

def execute(command, config_file=None, additional_configs=None):
    config = read_config_from_file(config_file)
    if additional_configs is not None:
        config.update(additional_configs)
    
    SCHEME = import_scheme(config['scheme'])
    
    world_size = torch.cuda.device_count()
    
    if 'random_seed' in config and config['random_seed'] is not None:
        seed = config['random_seed']
    else:
        seed = random.randint(0, 100000)
    
    if 'distributed' in config and config['distributed'] and world_size>1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'
        torch.multiprocessing.spawn(fn = run_worker,
                                    args = (world_size,command,SCHEME,config,seed),
                                    nprocs = world_size,
                                    join = True)
    else:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        scheme = SCHEME(config)
        if command == 'train':
            scheme.execute_training()
        elif command == 'predict':
            scheme.make_predictions()
        elif command == 'evaluate':
            scheme.do_evaluations()
        
