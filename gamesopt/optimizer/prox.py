from dataclasses import dataclass
from enum import Enum
from typing import Optional
import torch
import torch.nn.functional as F

class ProxType(Enum):
    NONE = "none"
    L1_REG = "L1_reg"
    LINF_BALL_L1_REG = "Linf-ball+L1-reg"

@dataclass
class ProxOptions:
    prox_type: ProxType = ProxType.NONE
    l1_reg: float = 1.
    ball_radius: float = 1.


def load_prox(options: ProxOptions = ProxOptions()):
    if options.prox_type == ProxType.NONE:
        return DefaultProx()
    elif options.prox_type == ProxType.L1_REG:
        return L1RegProx(options)
    elif options.prox_type == ProxType.LINF_BALL_L1_REG:
        return LinfBallL1RegProx(options)


class DefaultProx:
    def __call__(self, x: torch.Tensor, lr: Optional[float] = None) -> torch.Tensor:
        return x


class L1RegProx(DefaultProx):
    def __init__(self, options: ProxOptions=ProxOptions()) -> None:
        self.l1_reg = options.l1_reg

    def __call__(self, x: torch.Tensor, lr: Optional[float]) -> torch.Tensor:
        return  x.sign()*F.relu(abs(x) - lr*self.l1_reg)
       

class LinfBallL1RegProx(DefaultProx):
    def __init__(self, options: ProxOptions=ProxOptions()) -> None:
        self.l1_reg = options.l1_reg
        self.ball_radius = options.ball_radius

    def __call__(self, x: torch.Tensor, lr: Optional[float]) -> torch.Tensor:
        return  x.sign()*torch.minimum(F.relu(abs(x) - lr*self.l1_reg), torch.zeros_like(x) + self.ball_radius)
