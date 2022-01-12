from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
from gamesopt.games import Game
from enum import Enum
from dataclasses import dataclass
from .lr import LRScheduler, FixedLR
import torch


class OptimizerType(Enum):
    PROX_SGDA = "Prox-SGDA"
    PROX_LSVRGDA = "Prox-L-SVRGDA"
    SVRG = "SVRG"
    VRFORB = "VR-FoRB"
    VRAGDA = "VR-AGDA"
    SVRE = "svre"
    EG_VR = "EG-VR"

LRSchedulerType = Union[float, LRScheduler]

@dataclass
class OptimizerOptions:
    optimizer_type: OptimizerType = OptimizerType.PROX_SGDA
    lr:  Union[LRSchedulerType, Tuple[LRSchedulerType, ...]] = 1e-2
    lr_e:  Optional[Union[LRSchedulerType, Tuple[LRSchedulerType, ...]]] = None
    p: Optional[float] = None
    alpha: float = 0.
    full_batch: bool = False
    batch_size: int = 1
    N: int = 1
    T: int = 1

    def __post_init__(self):
        if isinstance(self.lr, float):
            self.lr = FixedLR(self.lr)
        if isinstance(self.lr_e, float):
            self.lr_e = FixedLR(self.lr_e)


class Optimizer(ABC):
    def __init__(self, game: Game, options: OptimizerOptions) -> None:
        self.game: Game = game
        self.k = 0
        self.lr = options.lr
        self.batch_size = options.batch_size
        self.full_batch = options.full_batch

    def sample(self) -> Optional[torch.Tensor]:
        if self.full_batch:
            return self.game.sample_batch()
        else:
            return self.game.sample(self.batch_size)

    @abstractmethod
    def step(self) -> None:
        pass