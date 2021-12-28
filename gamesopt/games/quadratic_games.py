from .base import Game
import torch
import math


def make_random_matrix(num_samples: int, dim: int) -> torch.Tensor:
    return torch.zeros(num_samples, dim, dim)

def random_vector(dim: int) -> torch.Tensor:
    x = torch.zeros(dim).normal_() / math.sqrt(dim)
    x.requires_grad_()
    return x

class QuadraticGame(Game):
    def __init__(self, num_samples: int, dim: int, num_players: int = 2, bias: bool = True) -> None:
        self.dim = dim
        self.matrix = make_random_matrix(num_samples, num_players*dim)
        self.bias = torch.zeros(2, num_samples, dim)
        if bias:
            self.bias = self.bias.normal_() / (10 * math.sqrt(self.dim))
        players = [torch.zeros(self.dim, requires_grad=True), torch.zeros(self.dim, requires_grad=True)]
        self.optimum = self.solve()

        super().__init__(players)

    def reset(self):
        for i in range(self.num_players):
            self.players[i] = random_vector(self.dim)

    def loss(self, index):
        loss =  []
        for i in range(self.num_players):
            _loss = self.bias[i, index]
            for j in range(self.num_players):
                _loss += (self.matrix[index, i*self.dim:(i+1)*self.dim, j*self.dim:(j+1)*self.dim]*self.players[j].view(1,1,-1)).sum(-1)
            _loss = self.players[i].view(1, -1).sum(-1).mean()
            loss.append(_loss)
        return loss

    def dist2opt(self):
        d = 0
        for i in range(self.num_players):
            d += (self.players[i] - self.optimum[i]) ** 2).sum()
        return d

    def solve(self):
        b = self.bias.mean(1).view(-1)
        sol = torch.linalg.solve(self.matrix.mean(0), -b)
        sol = sol.split(self.num_players)
        return sol