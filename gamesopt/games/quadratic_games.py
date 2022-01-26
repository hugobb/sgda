from pathlib import Path
from .base import Game
from .utils import random_vector
import torch
import math
from dataclasses import dataclass
from typing import Optional, Union, Tuple
from torch import linalg


def make_random_matrix(num_players: int, num_samples: int, dim: int, mu: float = 0, L: float = 1., max_im: float = 1.) -> torch.Tensor:
    if isinstance(L, float):
        L_min = L
        L_max = L
    elif len(L) == 2:
        L_min = L[0]
        L_max = L[1]
    else:
        raise ValueError()

    matrix = torch.randn(num_samples, num_players*dim, num_players*dim)
    L, V = torch.linalg.eig(matrix)
    L.real = abs(L.real)
    matrix = (V @ torch.diag_embed(L) @ torch.linalg.inv(V)).real
    #s = torch.linalg.eigvals(matrix)
    #print(s.real.max(dim=-1)[0].min(), s.real.max(dim=-1)[0].max(), s.real.min(), s.imag.max(), abs(s.imag).min())
    return matrix

@dataclass
class QuadraticGameConfig:
    num_samples: int = 10
    dim: int = 2
    num_players: int = 2
    bias: bool = True
    mu: float = 0.
    L: Union[float, Tuple[float, float]] = 1.
    max_im: float = 1.
    importance_sampling: bool = False


class QuadraticGame(Game):
    def __init__(self, config: QuadraticGameConfig = QuadraticGameConfig(), rank: Optional[int] = None) -> None:
        self.config = config
        self._dim = config.dim
        players = [torch.zeros(self._dim, requires_grad=True), torch.zeros(self._dim, requires_grad=True)]
        super().__init__(players, config.num_samples, rank)
        
        self.matrix = make_random_matrix(config.num_players, config.num_samples, config.dim, config.mu, config.L, config.max_im) 

        self.bias = torch.zeros(2, config.num_samples, config.dim)
        if config.bias:
            self.bias = self.bias.normal_() / (10 * math.sqrt(self._dim))

        self.p = torch.ones(self.num_samples) / self.num_samples
        self.importace_sampling = config.importance_sampling
        if self.importace_sampling:
            self.set_p()       

        self.reset()

    def set_p(self):
        eigenvalues: torch.Tensor = linalg.eigvals(self.matrix)
        ell = 1 / ((1 / eigenvalues).real).min(-1)[0]
        self.p = ell / (ell.sum())

    def reset(self) -> None:
        for i in range(self.num_players):
            self.players[i] = random_vector(self._dim)

    def sample(self, n: int = 1) -> torch.Tensor:      
        return torch.multinomial(self.p, n, replacement=True)

    def sample_batch(self) -> torch.Tensor:
        return torch.arange(self.num_samples).long()

    def loss(self, index: int) -> torch.Tensor:
        loss =  []
        for i in range(self.num_players):
            _loss = self.bias[i, index]
            for j in range(self.num_players):
                _loss += (self.matrix[index, i*self._dim:(i+1)*self._dim, j*self._dim:(j+1)*self._dim]*self.players[j].view(1,1,-1)).sum(-1)
            _loss = (_loss*self.players[i].view(1, -1)).sum(-1).mean()
            loss.append(_loss)
        return loss

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        filename = "model"
        if self.rank is not None:
            filename += "_%i"%self.rank
        filename = path / ("%s.pth" % filename)

        torch.save({"config": self.config, "players": self.players, "matrix": self.matrix, "bias": self.bias}, filename)

    def load(self, path: Path, copy: bool = False) -> Game:
        filename = "model"
        if self.rank is not None:
            filename += "_%i"%self.rank
        filename = path / ("%s.pth" % filename)

        checkpoint = torch.load(filename)
        self.matrix = checkpoint["matrix"]
        self.bias = checkpoint["bias"]
        if self.importace_sampling:
            self.set_p() 

        if copy:
            game = self.copy()
            game.players = checkpoint["players"]
            return game
        else:
            self.players = checkpoint["players"]
            return self

        
    

