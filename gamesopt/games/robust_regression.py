from typing import List, Optional
from .base import Game
from dataclasses import dataclass
import torch
from .utils import random_vector


@dataclass
class RobustLinRegConfig:
    num_samples: int = 10
    dim: int = 2
    lambda_coeff: float = 0.2
    gamma_coeff: float = 0.2


class RobustLinReg(Game):
    def __init__(self, config: RobustLinRegConfig = RobustLinRegConfig()) -> None:
        self.dim = config.dim
        players = [torch.zeros(self.dim, requires_grad=True), torch.zeros(self.dim, requires_grad=True)]

        super().__init__(players, config.num_samples)

        self.x, self.y = torch.zeros(self.num_samples, self.dim).normal_(), torch.zeros(self.num_samples).normal_()
        self.lambda_coeff = config.lambda_coeff
        self.gamma_coeff = config.gamma_coeff

    def reset(self) -> None:
        for i in range(self.num_players):
            self.players[i] = random_vector(self.dim)

    def loss(self, index: Optional[int] = None) -> List[torch.Tensor]:
        loss = 0.5*(
            ((self.players[0].view(1, -1)*(self.x[index] + self.players[1].view(1, -1))).sum(-1) - self.y[index])**2 
            + 0.5*self.lambda_coeff*(self.players[0]**2).sum()
            - 0.5*self.gamma_coeff*(self.players[1]**2).sum()
        ).mean()
        return [loss, -loss]

    def sample(self, n: int = 1) -> torch.Tensor:
        return torch.randint(self.num_samples, size=(n, ))

    def sample_batch(self) -> torch.Tensor:
        return torch.arange(self.num_samples).long()