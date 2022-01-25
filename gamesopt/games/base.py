from abc import ABC, abstractmethod
from pathlib import Path
import torch.autograd as autograd
import torch
from typing import List, Optional
import copy
import torch.distributed as dist


class Game(ABC):
    def __init__(self, players: List[torch.Tensor], num_samples: int) -> None:
        self.num_players = len(players)
        self.players = players
        self.num_samples = num_samples
        self.master_node = None
        self.n_process = 1

        self.shape = []
        self.split_size = []
        for p in self.players:
            self.shape.append(p.size())
            self.split_size.append(p.numel())

    def broadcast(self, src: int) -> None:
        for i in range(self.num_players):
            dist.broadcast(self.players[i], src)

    def unflatten(self, index: int, x: torch.Tensor) -> torch.Tensor:
        return x.split(self.split_size)[index].view(self.shape[index])

    def flatten(self, tensor_list: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat([ x.view(-1) for x in tensor_list])

    def set_master_node(self, rank: int, n_process: int):
        self.master_node = rank
        self.n_process = n_process

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

    def set_players(self, players):
        for i in range(self.num_players):
            self.players[i] = players[i].clone()

    def loss(self, index: Optional[int] = None):
        raise NotImplementedError("You need to overwrite either `loss` or `operator`, when inheriting `Game`.")

    def operator(self, index: Optional[int] = None, player_index: Optional[int] = None) -> torch.Tensor:
        loss = self.loss(index)
        if player_index is None:
            return self.flatten(map(self.grad, loss, range(self.num_players)))
        else:
            return self.grad(loss[player_index], player_index)

    def grad(self, loss: torch.Tensor, index: int) -> torch.Tensor:
        return autograd.grad(loss, self.players[index], retain_graph=True)[0]

    def full_operator(self) -> torch.Tensor:
        index = self.sample_batch()
        return self.operator(index)

    def hamiltonian(self) -> float:
        index = self.sample_batch()
        grad = self.operator(index)

        if self.master_node is not None:
            dist.reduce(grad, self.master_node)
            grad /= self.n_process

        hamiltonian = (grad**2).sum()
        hamiltonian /= 2

        return float(hamiltonian)

    def sample_batch(self) -> Optional[torch.Tensor]:
        return None

    def sample(self, n: int = 1) -> Optional[torch.Tensor]:
        return None

    def update_players(self, players: List[torch.Tensor]) -> None:
        self.players = players

    def save(self, filename: Path) -> None:
        pass

    @staticmethod
    def load(filename: Path):
        pass

    def dist(self, game) -> float:
        dist = 0
        for i in range(self.num_players):
            dist += ((game.players[i] - self.players[i])**2).sum()
        return float(dist)
        

