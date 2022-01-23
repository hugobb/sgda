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


@dataclass
class TrainDistributedConfig(TrainConfig):
    n_process: int = 2
    optimizer: OptimizerOptions = OptimizerOptions(optimizer_type=OptimizerType.QSGDA)

def _train(rank, config: TrainDistributedConfig = TrainDistributedConfig(), record: Record = Record()) -> None:
    setup(rank, config.n_process)
    
    game = load_game(config.game)
    game.set_master_node(0, config.n_process)
    game.broadcast(0)
    optimizer: DistributedOptimizer = load_optimizer(game, config.optimizer)

    metrics = defaultdict(list)
    for _ in range(config.num_iter):
        optimizer.step()

        hamiltonian = game.hamiltonian()
        num_grad = optimizer.get_num_grad()
        n_bits = optimizer.get_n_bits()

        if rank == 0:
            metrics["hamiltonian"].append(hamiltonian)
            metrics["num_grad"].append(num_grad)
            metrics["n_bits"].append(n_bits)
            record.save_metrics(metrics)

def setup(rank: int, size: int, backend: str = 'gloo') -> None:
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)


def train(config: TrainDistributedConfig = TrainDistributedConfig(), record: Record = Record()) -> Record:
        record.save_config(config)
        torch.manual_seed(config.seed)
        mp.spawn(_train, args=(config, record), nprocs=config.n_process, join=True)
        return record