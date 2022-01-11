from .base import Optimizer, OptimizerOptions
from gamesopt.games import Game
from typing import List, Optional


class ProxSGDA(Optimizer):
    def __init__(self, game: Game, options: OptimizerOptions = OptimizerOptions()) -> None:
        super().__init__(game, options)
        self.lr = options.lr

    def step(self, index: Optional[int] = None) -> None:
        grad = self.update.grad(index)
        
        for i, g in enumerate(grad):
            self.game.players[i] = self.game.prox(self.game.players[i] - self.lr(self.k)*g)
        
        self.update.update_state()

        self.k += 1

