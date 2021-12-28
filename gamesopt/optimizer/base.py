from abc import ABC, abstractmethod
from gamesopt.games import Game


class Optimizer(ABC):
    def __init__(self, game: Game) -> None:
        self.game = game

    @abstractmethod
    def step(self) -> None:
        pass