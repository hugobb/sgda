from typing import Callable, Optional
from .base import Optimizer
from .sgda import ProxSGDA, ProxSVRGDA
from gamesopt.games import Game
from enum import Enum
from dataclasses import dataclass
import torch


class OptimizerType(Enum):
    PROX_SGDA = "prox_sgda"
    PROX_SVRGDA = "prox_svrgda"


@dataclass
class OptimizerOptions:
    optimizer_type: OptimizerType = OptimizerType.PROX_SGDA
    lr: float = 1e-2
    p: Optional[float] = None


def load_optimizer(game: Game, options: OptimizerOptions = OptimizerOptions()) -> Optimizer:
    if options.optimizer_type == OptimizerType.PROX_SGDA:
        return ProxSGDA(game, options.lr)
    elif options.optimizer_type == OptimizerType.PROX_SVRGDA:
        return ProxSVRGDA(game, options.lr, options.p)