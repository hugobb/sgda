from gamesopt.db import Record
from .optimizer.base import DistributedOptimizer
from .games import load_game
from.optimizer import load_optimizer, OptimizerOptions, OptimizerType
from dataclasses import dataclass
from collections import defaultdict
from .train import TrainConfig
import torch.multiprocessing as mp
import torch.distributed as dist
import os
import torch
import uuid
import random


class PortNotAvailableError(Exception):
    pass

@dataclass
class TrainDistributedConfig(TrainConfig):
    n_process: int = 2
    optimizer: OptimizerOptions = OptimizerOptions(optimizer_type=OptimizerType.QSGDA)

def _train(rank: int, port: str, config: TrainDistributedConfig = TrainDistributedConfig(), record: Record = Record()) -> None:
    setup(rank, config.n_process, port)
    
    print("Init...")
    game = load_game(config.game, rank)
    game.set_master_node(0, config.n_process)
    game.broadcast(0)
    if config.load_file is not None:
        game_copy = game.load(config.load_file, copy=True)
    
    optimizer: DistributedOptimizer = load_optimizer(game, config.optimizer)

    print("Starting...")
    metrics = defaultdict(list)
    for _ in range(config.num_iter):
        if rank == 0:
            metrics["hamiltonian"].append(hamiltonian)
            metrics["num_grad"].append(num_grad)
            metrics["n_bits"].append(n_bits)
            record.save_metrics(metrics)

            if config.load_file:
                metrics["dist2opt"].append(game.dist(game_copy))
                
        optimizer.step()

        hamiltonian = game.hamiltonian()
        num_grad = optimizer.get_num_grad()
        n_bits = optimizer.get_n_bits()

        print(rank, hamiltonian)

        if config.save_file is not None:
            game.save(config.save_file)

def setup(rank: int, size: int, port: str, backend: str = 'gloo') -> None:
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = port
    try:
        dist.init_process_group(backend, rank=rank, world_size=size)
    except:
        raise PortNotAvailableError


def train(config: TrainDistributedConfig = TrainDistributedConfig(), record: Record = Record()) -> Record:
    record.save_config(config)
    torch.manual_seed(config.seed)

    # Tries to allocate a port until a port is available
    while True:
        port = str(random.randrange(1030, 49151))
        print("Trying port %s" % port)
        try:
            mp.spawn(_train, args=(port, config, record), nprocs=config.n_process, join=True)
            break
        except PortNotAvailableError:
            print("Port %s not available" % port)
        else:
            raise
    
    record.load_metrics()
    return record