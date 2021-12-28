from abc import ABC, abstractmethod
import torch.autograd as autograd
import torch
from typing import List


class Game(ABC):
    def __init__(self, players: List[torch.Tensor]) -> None:
        self.num_players = len(players)
        self.players = players

    def loss(self, index: int):
        raise NotImplementedError("You need to overwrite either `loss` or `operator`, when inheriting `Game`.")

    def operator(self, index: int) -> List[List[torch.Tensor]]:
        loss = self.loss(index)
        grad = []
        for i in range(self.num_players):
            _grad = autograd.grad(loss[i], self.players[i], retain_graph=True)
            grad.append(_grad)

        return grad

    def hamiltonian(self) -> int:
        index = torch.arange(self.num_samples).long()
        grad = self.operator(index)

        hamiltonian = 0
        for g in grad:
            for _g in g:
                hamiltonian += (_g**2).sum()
        hamiltonian /= 2

        return int(hamiltonian)

    @abstractmethod
    def sample(self, n: int = 1):
        pass
