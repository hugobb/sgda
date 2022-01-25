from .optimizer.prox import ProxOptions, load_prox
from .games import load_game, GameOptions
from .optimizer import load_optimizer, OptimizerOptions
from dataclasses import dataclass
from collections import defaultdict
import torch
from .db import Record
from pathlib import Path
from typing import Optional

@dataclass
class TrainConfig:
    game: GameOptions = GameOptions()
    optimizer: OptimizerOptions = OptimizerOptions()
    prox: ProxOptions = ProxOptions()
    num_iter: int = 100
    seed: int = 1234
    name: str = ""
    save_file: Optional[Path] = None
    load_file: Optional[Path] = None
    precision: float = 1.


def train(config: TrainConfig = TrainConfig(), record: Record = Record()) -> Record:
    record.save_config(config)
    torch.manual_seed(config.seed)
    
    print("Init...")
    game = load_game(config.game)
    if config.load_file is not None:
        game_copy = game.load(config.load_file, copy=True)
    
    prox = load_prox(config.prox)    
    optimizer = load_optimizer(game, config.optimizer, prox)

    
    metrics = defaultdict(list)
    for _ in range(config.num_iter):
        optimizer.step()
        
        metrics["hamiltonian"].append(game.hamiltonian())
        metrics["num_grad"].append(optimizer.num_grad)
        metrics["prox_dist"].append(optimizer.fixed_point_check(config.precision))
        if config.load_file:
            metrics["dist2opt"].append(game.dist(game_copy))
        
        record.save_metrics(metrics)
        
        if config.save_file is not None:
            game.save(config.save_file)

    return record