from abc import ABC


from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, game: Game) -> None:
        super().__init__()
        self.game = game

    @abstractmethod
    def update(self) -> None:
        pass

    def step(self) -> None:
        update = self.update()
        self.game.update_parameters(update)