from .optimizer.base import DistributedOptimizer
from .games import load_game
from.optimizer import load_optimizer, OptimizerOptions, OptimizerType
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List
from omegaconf import OmegaConf
from .train import TrainConfig
import torch.multiprocessing as mp
import torch.distributed as dist
import os
import torch
import tempfile


@dataclass
class TrainDistributedConfig(TrainConfig):
    n_process: int = 2
    optimizer: OptimizerOptions = OptimizerOptions(optimizer_type=OptimizerType.QSGDA)

def _train(rank, config: TrainDistributedConfig = TrainDistributedConfig(), tmp_file = None) -> Dict[str, List[float]]:
    setup(rank, config.n_process)
    
    game = load_game(config.game)
    game.set_master_node(0, config.n_process)
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
            conf = OmegaConf.create(dict(metrics))
            OmegaConf.save(config=conf, f=tmp_file)

def setup(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)


def train(config: TrainDistributedConfig = TrainDistributedConfig(), save_tmp: bool = False):
    with tempfile.NamedTemporaryFile() as fp:
        torch.manual_seed(config.seed)
        mp.spawn(_train, args=(config, fp.name), nprocs=config.n_process, join=True)
        return OmegaConf.load(fp.name)