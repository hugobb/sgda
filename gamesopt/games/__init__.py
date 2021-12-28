from .base import Game
from .quadratic_games import QuadraticGame, QuadraticGameConfig
from enum import Enum
from dataclasses import dataclass


class GameType(Enum):
    QUADRATIC = "quadratic"


@dataclass
class GameOptions:
    game_type: GameType = GameType.QUADRATIC
    quadratic_options: QuadraticGameConfig = QuadraticGameConfig()


def load_game(options: GameOptions = GameOptions()) -> Game:
    if options.game_type == GameType.QUADRATIC:
        return QuadraticGame(options.quadratic_options)

