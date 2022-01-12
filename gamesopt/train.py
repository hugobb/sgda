from .games import load_game, GameOptions
from.optimizer import load_optimizer, OptimizerOptions
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List
import torch

@dataclass
class TrainConfig:
    game: GameOptions = GameOptions()
    optimizer: OptimizerOptions = OptimizerOptions()
    num_iter: int = 100
    seed: int = 1234


def train(config: TrainConfig = TrainConfig()) -> Dict[str, List[float]]:
    torch.manual_seed(config.seed)
    
    game = load_game(config.game)
    optimizer = load_optimizer(game, config.optimizer)

    metrics = defaultdict(list)
    for _ in range(config.num_iter):
        optimizer.step()
        metrics["hamiltonian"].append(game.hamiltonian())

    return metrics