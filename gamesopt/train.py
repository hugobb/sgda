from .games import load_game, GameOptions
from.optimizer import load_optimizer, OptimizerOptions
from dataclasses import dataclass

@dataclass
class TrainConfig:
    game: GameOptions = GameOptions()
    optimizer: OptimizerOptions = OptimizerOptions()
    num_iter: int = 100
    batch_size: int = 1


def train(config: TrainConfig):
    game = load_game(config.game)
    optimizer = load_optimizer(game, config.optimizer)

    for _ in range(config.num_iter):
        index = game.sample(config.batch_size)
        optimizer.step(index)