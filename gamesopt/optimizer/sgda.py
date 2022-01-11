import torch
from torch.functional import Tensor
from .base import Optimizer
from gamesopt.games import Game
from typing import Callable, Optional
from copy import deepcopy


class ProxSGDA(Optimizer):
    def __init__(self, game: Game, lr: float = 1e-2, prox: Callable[[torch.Tensor], torch.Tensor] = lambda x : x) -> None:
        super().__init__(game)
        self.lr = lr

        self.prox = prox

    def step(self, index: Optional[int] = None) -> None:
        grad = self.game.operator(index)
        
        for i in range(self.game.num_players):
            self.game.players[i] = self.prox(self.game.players[i] - self.lr*grad[i])


class ProxSVRGDA(ProxSGDA):
    def __init__(self, game: Game, lr: float = 1e-2, p: Optional[float] = None, prox: Callable[[torch.Tensor], torch.Tensor] = lambda x : x) -> None:
        super().__init__(game, lr, prox)

        if p is None:
            p = 1/game.num_samples
        self.p = torch.as_tensor(p)

        self.update_state()

    def step(self, index: Optional[int] = None) -> None:
        grad = self.game.operator(index)
        grad_copy = self.game_copy.operator(index)
        
        for i in range(self.game.num_players):
            update = grad[i] - grad_copy[i] + self.full_grad[i]
            self.game.players[i] = self.prox(self.game.players[i] - self.lr*update)

        if not self.p.bernouilli_():
            self.update_state()

    def update_state(self) -> None:
        self.game_copy = deepcopy(self.game)
        self.full_grad = self.game_copy.full_operator()
