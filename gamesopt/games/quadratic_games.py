from .base import Game
from .utils import random_vector
import torch
import math
from dataclasses import dataclass
from typing import List, Optional


def make_random_matrix(num_samples: int, dim: int, mu: float = 0, L: float = 1., max_im: float = 1.) -> torch.Tensor:
    if isinstance(L, float):
        L_min = L
        L_max = L
    elif len(L) == 2:
        L_min = L[0]
        L_max = L[1]
    else:
        raise ValueError()

    matrix = torch.randn(num_samples, dim, dim)
    _, matrix = torch.linalg.eig(matrix)
    
    L_i = torch.rand(num_samples, 1)
    L_i = (L_i - L_i.min()) / (L_i.max() - L_i.min())
    L_i = L_min + L_i * (L_max - L_min)

    real_part = torch.rand(num_samples, dim)
    real_part = (real_part - real_part.min()) / (real_part.max() - real_part.min())
    real_part = mu + real_part * (L_i - mu)
    
    im_part = torch.rand(num_samples, dim)
    im_part = (im_part - im_part.min()) / (im_part.max() - im_part.min())
    im_part = (2*im_part - 1)*max_im

    eigs = torch.complex(real_part, im_part)

    matrix = torch.matmul(matrix, torch.matmul(eigs.diag_embed(), matrix.inverse())).real
    matrix[:, :dim, :dim] = 0.5*(matrix[:, :dim, :dim].transpose(-1, -2) + matrix[:, :dim, :dim])
    matrix[:, dim:, dim:] = 0.5*(matrix[:, dim:, dim:].transpose(-1, -2) + matrix[:, dim:, dim:])
    
    s = torch.linalg.eigvals(matrix)
    return matrix


@dataclass
class QuadraticGameConfig:
    num_samples: int = 10
    dim: int = 2
    num_players: int = 2
    bias: bool = True
    matrix: Optional[torch.Tensor] = None


class QuadraticGame(Game):
    def __init__(self, config: QuadraticGameConfig = QuadraticGameConfig()) -> None:
        self.dim = config.dim
        players = [torch.zeros(self.dim, requires_grad=True), torch.zeros(self.dim, requires_grad=True)]
        super().__init__(players, config.num_samples)
        
        if config.matrix is None:
            self.matrix = make_random_matrix(config.num_samples, config.num_players*config.dim) 
        else:
            self.matrix = config.matrix

        self.bias = torch.zeros(2, config.num_samples, config.dim)
        if config.bias:
            self.bias = self.bias.normal_() / (10 * math.sqrt(self.dim))

        self.optimum = self.solve()

        self.reset()   

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
            _loss = (_loss*self.players[i].view(1, -1)).sum(-1).mean()
            loss.append(_loss)
        return loss

    def dist2opt(self) -> float:
        d = 0
        for i in range(self.num_players):
            d += ((self.players[i] - self.optimum[i]) ** 2).sum()
        return float(d)

    def solve(self) -> List[torch.Tensor]:
        b = torch.cat([self.bias[0], self.bias[1]], dim=-1).mean(0)
        sol = torch.linalg.solve(self.matrix.mean(0), -b)
        sol = torch.split(sol, self.num_players)
        return sol