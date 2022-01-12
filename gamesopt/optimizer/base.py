from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
from gamesopt.games import Game
from enum import Enum
from dataclasses import dataclass
from .lr import LRScheduler, FixedLR
from .vr import load_update_scheme, UpdateSchemeOptions


class OptimizerType(Enum):
    PROX_SGDA = "prox_sgda"
    EXTRAGRADIENT = "extragradient"
    EG_WITH_VR = "eg_with_vr"


LRSchedulerType = Union[float, LRScheduler]

@dataclass
class OptimizerOptions:
    optimizer_type: OptimizerType = OptimizerType.PROX_SGDA
    lr:  Union[LRSchedulerType, Tuple[LRSchedulerType, ...]] = 1e-2
    lr_e:  Optional[Union[LRSchedulerType, Tuple[LRSchedulerType, ...]]] = None
    update_scheme_options: UpdateSchemeOptions = UpdateSchemeOptions()
    alpha: float = 0.

    def __post_init__(self):
        if isinstance(self.lr, float):
            self.lr = FixedLR(self.lr)
        if isinstance(self.lr_e, float):
            self.lr_e = FixedLR(self.lr_e)


class Optimizer(ABC):
    def __init__(self, game: Game, options: OptimizerOptions) -> None:
        self.game = game
        self.k = 0

        self.update = load_update_scheme(game, options.update_scheme_options) 

    @abstractmethod
    def step(self, index: Optional[int] = None) -> None:
        pass