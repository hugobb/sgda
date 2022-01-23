from .games import load_game, GameOptions
from .optimizer import load_optimizer, OptimizerOptions
from dataclasses import dataclass
from collections import defaultdict
import torch
from .db import Record

@dataclass
class TrainConfig:
    game: GameOptions = GameOptions()
    optimizer: OptimizerOptions = OptimizerOptions()
    num_iter: int = 100
    seed: int = 1234
    name: str = ""


def train(config: TrainConfig = TrainConfig(), record: Record = Record()) -> Record:
    record.save_config(config)
    torch.manual_seed(config.seed)
    game = load_game(config.game)
    optimizer = load_optimizer(game, config.optimizer)

    metrics = defaultdict(list)
    for _ in range(config.num_iter):
        optimizer.step()
        metrics["hamiltonian"].append(game.hamiltonian())
        metrics["num_grad"].append(optimizer.num_grad)
        record.save_metrics(metrics)

    return record