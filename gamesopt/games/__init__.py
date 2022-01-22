from .bilinear import BilinearGame, BilinearGameConfig
from .robust_regression import RobustLinRegConfig, RobustLinReg, RobustLogReg
from .base import Game
from .quadratic_games import QuadraticGame, QuadraticGameConfig
from .kelly_auction import KellyAuction, KellyAuctionConfig
from enum import Enum
from dataclasses import dataclass


class GameType(Enum):
    QUADRATIC = "quadratic"
    KELLY_AUCTION = "kelly_auction"
    ROBUST_LINEAR_REG = "robust_linear_reg"
    BILINEAR = "bilinear"
    ROBUST_LOGISTIC_REG = "robust_logistic_regression"


@dataclass
class GameOptions:
    game_type: GameType = GameType.QUADRATIC
    quadratic_options: QuadraticGameConfig = QuadraticGameConfig()
    kelly_auction_options: KellyAuctionConfig = KellyAuctionConfig()
    robust_linear_reg_options: RobustLinRegConfig = RobustLinRegConfig()
    bilinear_options: BilinearGameConfig = BilinearGameConfig()


def load_game(options: GameOptions = GameOptions()) -> Game:
    if options.game_type == GameType.QUADRATIC:
        return QuadraticGame(options.quadratic_options)
    elif options.game_type == GameType.KELLY_AUCTION:
        return KellyAuction(options.kelly_auction_options)
    elif options.game_type == GameType.ROBUST_LINEAR_REG:
        return RobustLinReg(options.robust_linear_reg_options)
    elif options.game_type == GameType.BILINEAR:
        return BilinearGame(options.bilinear_options)
    elif options.game_type == GameType.ROBUST_LOGISTIC_REG:
        return RobustLogReg()
    else:
        raise ValueError()

