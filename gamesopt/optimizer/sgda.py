from .base import Optimizer
from gamesopt.games import Game
from typing import Optional


class SGDA(Optimizer):
    def __init__(self, game: Game, lr: float = 1e-2) -> None:
        super().__init__(game)
        self.lr = lr

    def step(self, index: Optional[int] = None) -> None:
        grad = self.game.operator(index)
        
        for i in range(self.game.num_players):
            self.game.players[i] = self.game.players[i] - self.lr*grad[i]
            