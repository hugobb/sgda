from .base import Optimizer
from .sgda import SGDA
from gamesopt.games import Game
from enum import Enum
from dataclasses import dataclass

class OptimizerType(Enum):
    SGDA = "sgda"

@dataclass
class OptimizerOptions:
    optimizer_type: OptimizerType = OptimizerType.SGDA
    lr: float = 1e-2

def load_optimizer(game: Game, options: OptimizerOptions = OptimizerOptions()) -> Optimizer:
    if options.optimizer_type == OptimizerType.SGDA:
        return SGDA(game, options.lr)