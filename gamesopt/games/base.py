from abc import ABC, abstractmethod
from collections.abc import Iterator
import torch.autograd as autograd
import torch
from typing import List, Optional, Union
import copy


class Game(ABC):
    def __init__(self, players: List[torch.Tensor], num_samples: int) -> None:
        self.num_players = len(players)
        self.players = players
        self.num_samples = num_samples

    @abstractmethod
    def reset(self) -> None:
        pass

    def copy(self):
        players = []
        for i in range(self.num_players):
            players.append(self.players[i].clone())
        game = copy.copy(self)
        game.players = players
        return game

    def loss(self, index: Optional[int] = None):
        raise NotImplementedError("You need to overwrite either `loss` or `operator`, when inheriting `Game`.")

    def operator(self, index: Optional[int] = None, player_index: Optional[int] = None) -> Union[torch.Tensor, Iterator[torch.Tensor]]:
        loss = self.loss(index)
        if player_index is None:
            return map(self.grad, loss, range(self.num_players))
        else:
            return self.grad(loss[player_index], player_index)

    def grad(self, loss: torch.Tensor, index: int) -> torch.Tensor:
        return autograd.grad(loss, self.players[index], retain_graph=True)[0]

    def full_operator(self) -> List[List[torch.Tensor]]:
        index = self.sample_batch()
        return self.operator(index)

    def hamiltonian(self) -> float:
        index = self.sample_batch()
        grad = self.operator(index)

        hamiltonian = 0
        for g in grad:
            hamiltonian += (g**2).sum()
        hamiltonian /= 2

        return float(hamiltonian)

    def sample_batch(self) -> Optional[torch.Tensor]:
        return None

    def sample(self, n: int = 1) -> Optional[torch.Tensor]:
        return None

    def prox(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def update_players(self, players: List[torch.Tensor]) -> None:
        self.players = players
