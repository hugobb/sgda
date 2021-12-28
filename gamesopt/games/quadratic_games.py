from .base import Game
import torch
import math
from dataclasses import dataclass
from typing import List


def make_random_matrix(num_samples: int, dim: int) -> torch.Tensor:
    return torch.zeros(num_samples, dim, dim).normal_()

def random_vector(dim: int) -> torch.Tensor:
    x = torch.zeros(dim).normal_() / math.sqrt(dim)
    x.requires_grad_()
    return x


@dataclass
class QuadraticGameConfig:
    num_samples: int = 10
    dim: int = 2
    num_players: int = 2
    bias: bool = True


class QuadraticGame(Game):
    def __init__(self, config: QuadraticGameConfig = QuadraticGameConfig()) -> None:
        self.dim = config.dim
        players = [torch.zeros(self.dim, requires_grad=True), torch.zeros(self.dim, requires_grad=True)]
        super().__init__(players)

        self.num_samples = config.num_samples
        self.matrix = make_random_matrix(config.num_samples, config.num_players*config.dim)
        self.bias = torch.zeros(2, config.num_samples, config.dim)
        if config.bias:
            self.bias = self.bias.normal_() / (10 * math.sqrt(self.dim))
        self.optimum = self.solve()   

    def reset(self) -> None:
        for i in range(self.num_players):
            self.players[i] = random_vector(self.dim)

    def sample(self, n: int = 1) -> torch.Tensor:
        return torch.randint(self.num_samples, size=(n, ))

    def sample_batch(self) -> torch.Tensor:
        return torch.arange(self.num_samples).long()

    def loss(self, index: int) -> torch.Tensor:
        loss =  []
        for i in range(self.num_players):
            _loss = self.bias[i, index]
            for j in range(self.num_players):
                _loss += (self.matrix[index, i*self.dim:(i+1)*self.dim, j*self.dim:(j+1)*self.dim]*self.players[j].view(1,1,-1)).sum(-1)
            _loss = self.players[i].view(1, -1).sum(-1).mean()
            loss.append(_loss)
        return loss

    def dist2opt(self) -> int:
        d = 0
        for i in range(self.num_players):
            d += ((self.players[i] - self.optimum[i]) ** 2).sum()
        return int(d)

    def solve(self) -> List[torch.Tensor]:
        b = self.bias.mean(1).view(-1)
        sol = torch.linalg.solve(self.matrix.mean(0), -b)
        sol = sol.split(self.num_players)
        return sol