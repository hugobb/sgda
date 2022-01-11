
from .base import Optimizer, OptimizerOptions, OptimizerType
from .sgda import ProxSGDA
from gamesopt.games import Game


def load_optimizer(game: Game, options: OptimizerOptions = OptimizerOptions()) -> Optimizer:
    if options.optimizer_type == OptimizerType.PROX_SGDA:
        return ProxSGDA(game, options)