from abc import ABC, abstractmethod
import torch.autograd as autograd
import torch
from typing import List, Optional


class Game(ABC):
    def __init__(self, players: List[torch.Tensor]) -> None:
        self.num_players = len(players)
        self.players = players

    def loss(self, index: Optional[int] = None):
        raise NotImplementedError("You need to overwrite either `loss` or `operator`, when inheriting `Game`.")

    def operator(self, index: Optional[int] = None) -> List[List[torch.Tensor]]:
        loss = self.loss(index)
        grad = []
        for i in range(self.num_players):
            _grad = autograd.grad(loss[i], self.players[i], retain_graph=True)[0]
            grad.append(_grad)

        return grad

    def hamiltonian(self) -> int:
        index = self.sample_batch()
        grad = self.operator(index)

        hamiltonian = 0
        for g in grad:
            hamiltonian += (g**2).sum()
        hamiltonian /= 2

        return int(hamiltonian)

    def sample_batch(self) -> None:
        return None

    def sample(self, n: int = 1) -> None:
        return None
