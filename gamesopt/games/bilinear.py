from .base import Game
from .utils import random_vector
import torch
import math
from dataclasses import dataclass
from typing import List


@dataclass
class BilinearGameConfig:
    num_samples: int = 10
    dim: int = 2
    bias: bool = True


class BilinearGame(Game):
    def __init__(self, config: BilinearGameConfig = BilinearGameConfig()) -> None:
        self.dim = config.dim
        players = [torch.zeros(self.dim, requires_grad=True), torch.zeros(self.dim, requires_grad=True)]
        super().__init__(players, config.num_samples)
        
        self.matrix = torch.randn(config.num_samples, config.dim, config.dim)
        
        self.bias = torch.zeros(2, config.num_samples, config.dim)
        if config.bias:
            self.bias = self.bias.normal_() / (10 * math.sqrt(self.dim))

        self.x_star, self.y_star = self.solve()

        self.reset()   

    def reset(self) -> None:
        for i in range(self.num_players):
            self.players[i] = random_vector(self.dim)

    def sample(self, n: int = 1) -> torch.Tensor:
        return torch.randint(self.num_samples, size=(n, ))

    def sample_batch(self) -> torch.Tensor:
        return torch.arange(self.num_samples).long()

    def loss(self, index: int) -> torch.Tensor:
        loss = (self.players[0].view(1, -1) * (self.matrix[index] * self.players[1].view(1, 1, -1)).sum(-1)
                + self.bias[0, index] * self.players[0].view(1, -1)
                + self.bias[1, index] * self.players[1].view(1, -1)).sum(-1).mean()
        return [loss, -loss]

    def dist2opt(self) -> float:
        d = ((self.players[0] - self.x_star) ** 2).sum() + (
            (self.players[1] - self.y_star) ** 2
        ).sum()
        return float(d)

    def solve(self) -> List[torch.Tensor]:
        matrix = self.matrix.mean(0)
        x = torch.linalg.solve(matrix.T, -self.bias[1].mean(0))
        y = torch.linalg.solve(matrix, -self.bias[0].mean(0))

        return (x, y)