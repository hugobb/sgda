
from gamesopt.optimizer.distributed import DIANA_SGDA, QSGDA, VR_DIANA_SGDA
from .extragradient import EGwithVR, SVRE
from .base import Optimizer, OptimizerOptions, OptimizerType
from .sgda import SVRG, VRAGDA, ProxLSVRGDA, ProxSGDA, VRFoRB
from gamesopt.games import Game


def load_optimizer(game: Game, options: OptimizerOptions = OptimizerOptions()) -> Optimizer:
    if options.optimizer_type == OptimizerType.PROX_SGDA:
        return ProxSGDA(game, options)
    elif options.optimizer_type == OptimizerType.PROX_LSVRGDA:
        return ProxLSVRGDA(game, options)
    elif options.optimizer_type == OptimizerType.SVRG:
        return SVRG(game, options)
    elif options.optimizer_type == OptimizerType.VRAGDA:
        return VRAGDA(game, options)
    elif options.optimizer_type == OptimizerType.VRFORB:
        return VRFoRB(game, options)
    elif options.optimizer_type == OptimizerType.SVRE:
        return SVRE(game, options)
    elif options.optimizer_type == OptimizerType.EG_VR:
        return EGwithVR(game, options)
    elif options.optimizer_type == OptimizerType.QSGDA:
        return QSGDA(game, options)
    elif options.optimizer_type == OptimizerType.DIANA_SGDA:
        return DIANA_SGDA(game, options)
    elif options.optimizer_type == OptimizerType.VR_DIANA_SGDA:
        return VR_DIANA_SGDA(game, options)