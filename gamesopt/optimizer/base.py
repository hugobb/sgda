from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
from gamesopt.games import Game
from enum import Enum
from dataclasses import dataclass

from .prox import Prox
from .quantization import QuantizationOptions, load_quantization
from .lr import LRScheduler, FixedLR
import torch
import torch.distributed as dist


class OptimizerType(Enum):
    PROX_SGDA = "Prox-SGDA"
    PROX_LSVRGDA = "Prox-L-SVRGDA"
    SVRG = "SVRG"
    VRFORB = "VR-FoRB"
    VRAGDA = "VR-AGDA"
    SVRE = "SVRE"
    EG_VR = "EG-VR"
    QSGDA = "QSGDA"
    DIANA_SGDA = "DIANA-SGDA"
    VR_DIANA_SGDA = "VR-DIANA-SGDA"

LRSchedulerType = Union[float, LRScheduler]

@dataclass
class OptimizerOptions:
    optimizer_type: OptimizerType = OptimizerType.PROX_SGDA
    lr:  Union[LRSchedulerType, Tuple[LRSchedulerType, ...]] = 1e-2
    lr_e:  Optional[Union[LRSchedulerType, Tuple[LRSchedulerType, ...]]] = None
    p: Optional[float] = None
    alpha: Optional[float] = None
    full_batch: bool = False
    batch_size: int = 1
    N: Optional[int] = None
    T: int = 1
    quantization_options: QuantizationOptions = QuantizationOptions()

    def __post_init__(self):
        if isinstance(self.lr, float):
            self.lr = FixedLR(self.lr)
        if isinstance(self.lr_e, float):
            self.lr_e = FixedLR(self.lr_e)


class Optimizer(ABC):
    def __init__(self, game: Game, options: OptimizerOptions = OptimizerOptions(), prox: Prox = Prox()) -> None:
        self.game: Game = game
        self.k = 0
        self.lr = options.lr
        self.batch_size = options.batch_size
        self.full_batch = options.full_batch
        self.num_grad = 0

        self.prox = prox
        self.quantization = load_quantization(options.quantization_options)

        if isinstance(self.lr, float):
            self.lr = FixedLR(self.lr)

    def sample(self) -> Optional[torch.Tensor]:
        if self.full_batch:
            return self.game.sample_batch()
        else:
            return self.game.sample(self.batch_size)

    @abstractmethod
    def step(self) -> None:
        pass

    def fixed_point_check(self, precision: float = 1.) -> float:
        grad = self.game.full_operator()
        dist = 0
        for i in range(self.game.num_players):
            g = self.game.unflatten(i, grad)
            dist += ((self.game.players[i] - self.prox(self.game.players[i] - precision*g, precision))**2).sum()
        return float(dist)

class DistributedOptimizer(Optimizer):
    def __init__(self, game: Game, options: OptimizerOptions = OptimizerOptions(), prox: Prox = Prox()) -> None:
        super().__init__(game, options, prox)
        self.size = int(dist.get_world_size())
        self.n_bits = 0
    
    def get_num_grad(self) -> int:
        num_grad = torch.tensor([self.num_grad])
        dist.all_reduce(num_grad)
        return int(num_grad)

    def get_n_bits(self) -> int:
        n_bits = torch.tensor([self.n_bits])
        dist.all_reduce(n_bits)
        return int(n_bits)