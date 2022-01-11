from .vr import SVRG
from .base import Optimizer
from gamesopt.games import Game
from .lr import LRScheduler
from typing import Optional


class Extragradient(Optimizer):
    def __init__(self, game: Game, lr: LRScheduler, lr_e: Optional[LRScheduler] = None) -> None:
        super().__init__(game)
        self.lr = lr
        self.lr_e = lr_e
        if self.lr_e is None:
            self.lr_e = lr

    def step(self, index: Optional[int] = None) -> None:
        game_copy = self.game.copy()
        grad = self.game.operator(index)
        for i, g in enumerate(grad):
            self.game.players[i] = self.game.players[i] - self.lr_e(self.k)*g

        grad = self.game.operator(index)
        for i in range(self.game.num_players):
            self.game.players[i] = game_copy.players[i] - self.lr(self.k)*grad[i]
    
        self.k += 1


class SVRE(Optimizer):
    def __init__(self, game: Game, lr: LRScheduler, lr_e: Optional[LRScheduler] = None) -> None:
        super().__init__(game)
        self.lr = lr
        self.lr_e = lr_e
        if self.lr_e is None:
            self.lr_e = lr

        self.vr = SVRG(game)

    def step(self, index: Optional[int] = None) -> None:
        game_copy = self.game.copy()
        grad = self.game.operator(index)
        update = self.vr.update(grad)
        
        for i in range(self.game.num_players):
            self.game.players[i] = self.game.players[i] - self.lr_e(self.k)*grad[i]


        grad = self.game.operator(index)
        for i in range(self.game.num_players):
            self.game.players[i] = game_copy.players[i] - self.lr(self.k)*grad[i]

        self.vr.set_state(self.game)
    
        self.k += 1
