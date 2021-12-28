from .games import load_game, GameOptions
from.optimizer import load_optimizer, OptimizerOptions
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List

@dataclass
class TrainConfig:
    game: GameOptions = GameOptions()
    optimizer: OptimizerOptions = OptimizerOptions()
    num_iter: int = 100
    batch_size: int = 1
    full_batch: bool = False


def train(config: TrainConfig = TrainConfig()) -> Dict[str, List[float]]:
    game = load_game(config.game)
    optimizer = load_optimizer(game, config.optimizer)

    metrics = defaultdict(list)
    for _ in range(config.num_iter):
        if config.full_batch:
            index = game.sample_batch()
        else:
            index = game.sample(config.batch_size)
        optimizer.step(index)
        metrics["hamiltonian"].append(game.hamiltonian())
        metrics["dist2opt"].append(game.dist2opt())

    return metrics