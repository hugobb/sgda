from gamesopt.optimizer.lr import LRScheduler
import torch
from torch.functional import Tensor
from .base import Optimizer
from gamesopt.games import Game
from typing import Optional


class ProxSGDA(Optimizer):
    def __init__(self, game: Game, lr: LRScheduler) -> None:
        super().__init__(game)
        self.lr = lr

    def step(self, index: Optional[int] = None) -> None:
        grad = self.game.operator(index)
        
        for i in range(self.game.num_players):
            self.game.players[i] = self.game.prox(self.game.players[i] - self.lr(self.k)*grad[i])
        
        self.k += 1


class ProxSVRGDA(ProxSGDA):
    def __init__(self, game: Game, lr: LRScheduler, p: Optional[float] = None) -> None:
        super().__init__(game, lr)

        if p is None:
            p = 1/game.num_samples
        self.p = torch.as_tensor(p)

        self.update_state()

    def step(self, index: Optional[int] = None) -> None:
        grad = self.game.operator(index)
        grad_copy = self.game_copy.operator(index)
        
        for i in range(self.game.num_players):
            update = grad[i] - grad_copy[i] + self.full_grad[i]
            self.game.players[i] = self.game.prox(self.game.players[i] - self.lr(self.k)*update)

        if not self.p.bernoulli_():
            self.update_state()

        self.k += 1

    def update_state(self) -> None:
        self.game_copy = self.game.copy()
        self.full_grad = self.game_copy.full_operator()
