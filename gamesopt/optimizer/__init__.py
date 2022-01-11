from typing import Optional, Union
from .lr import FixedLR, LRScheduler
from .base import Optimizer
from .sgda import ProxSGDA, ProxSVRGDA
from gamesopt.games import Game
from enum import Enum
from dataclasses import dataclass


class OptimizerType(Enum):
    PROX_SGDA = "prox_sgda"
    PROX_SVRGDA = "prox_svrgda"


@dataclass
class OptimizerOptions:
    optimizer_type: OptimizerType = OptimizerType.PROX_SGDA
    lr: Union[float, LRScheduler] = 1e-2
    p: Optional[float] = None

    def __post_init__(self):
        if isinstance(self.lr, float):
            self.lr = FixedLR(self.lr)


def load_optimizer(game: Game, options: OptimizerOptions = OptimizerOptions()) -> Optimizer:
    if options.optimizer_type == OptimizerType.PROX_SGDA:
        return ProxSGDA(game, options.lr)
    elif options.optimizer_type == OptimizerType.PROX_SVRGDA:
        return ProxSVRGDA(game, options.lr, options.p)