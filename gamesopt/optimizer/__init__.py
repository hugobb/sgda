
from .extragradient import EGwithVR, Extragradient
from .base import Optimizer, OptimizerOptions, OptimizerType
from .sgda import ProxSGDA
from gamesopt.games import Game


def load_optimizer(game: Game, options: OptimizerOptions = OptimizerOptions()) -> Optimizer:
    if options.optimizer_type == OptimizerType.PROX_SGDA:
        return ProxSGDA(game, options)
    elif options.optimizer_type == OptimizerType.EXTRAGRADIENT:
        return Extragradient(game, options)
    elif options.optimizer_type == OptimizerType.EG_WITH_VR:
        return EGwithVR(game, options)