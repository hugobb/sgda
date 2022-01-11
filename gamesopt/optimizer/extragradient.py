from .base import Optimizer, OptimizerOptions
from gamesopt.games import Game
from .lr import LRScheduler
from typing import Optional


class Extragradient(Optimizer):
    def __init__(self, game: Game, options: OptimizerOptions = OptimizerOptions()) -> None:
        super().__init__(game, options)
        self.lr = options.lr

        if isinstance(self.lr, LRScheduler):
            self.lr = (self.lr,) * game.num_players

        self.lr_e = options.lr_e
        if self.lr_e is None:
            self.lr_e = self.lr

        if isinstance(self.lr, LRScheduler):
            self.lr = (self.lr,) * game.num_players
        

    def step(self, index: Optional[int] = None) -> None:
        game_copy = self.game.copy()
        grad = self.update.grad(index)
        for i, g in enumerate(grad):
            self.game.players[i] = self.game.players[i] - self.lr_e[i](self.k)*g

        grad = self.update.grad(index)
        for i, g in enumerate(grad):
            self.game.players[i] = game_copy.players[i] - self.lr[i](self.k)*g
    
        self.update.update_state()

        self.k += 1
