from dataclasses import dataclass
import torch
from gamesopt.games import Game
from typing import List, Optional
from enum import Enum

class UpdateType(Enum):
    GRADIENT = "gradient"
    SVRG = "SVRG"
    L_SVRG = "L-SVRG"
    VR_FORB = "VR-FoRB"


@dataclass
class UpdateSchemeOptions:
    update_scheme: UpdateType = UpdateType.GRADIENT
    p: Optional[float] = None


def load_update_scheme(game: Game, options: UpdateSchemeOptions):
    if options.update_scheme == UpdateType.GRADIENT:
        return GradientUpdate(game)
    elif options.update_scheme == UpdateType.SVRG:
        return SVRG(game)
    elif options.update_scheme == UpdateType.L_SVRG:
        return LooplessSVRG(game, options.p)
    elif options.update_scheme == UpdateType.VR_FORB:
        return VRFoRB(game)


class GradientUpdate:
    def __init__(self, game: Game) -> None:
        self.game = game
        self.game_copy = None
        self.full_grad = None

    def _update_state(self) -> None:
        pass

    def update_state(self) -> None:
        pass

    def grad(self, index: int) -> List[torch.Tensor]:
        return self.game.operator(index)


class SVRG(GradientUpdate):
    def __init__(self, game: Game) -> None:
        super().__init__(game)
        self.p = 1 / game.num_samples
        self.N = torch.tensor([0])
        self.k = 0
        self._update_state()

    def _update_state(self) -> None:
        self.game_copy = self.game.copy()
        self.full_grad = self.game_copy.full_operator()

    def update_state(self) -> None:
        if self.k >= self.N - 1:
            self._update_state()
            self.N.geometric_(self.p)
            self.k = 0
        self.k += 1

    def _grad(self, grad, grad_copy, full_grad):
        return grad - grad_copy + full_grad

    def grad(self, index: int) -> List[torch.Tensor]:
        grad = super().grad(index)
        grad_copy = self.game_copy.operator(index)

        return map(self._grad, grad, grad_copy, self.full_grad)
        

class LooplessSVRG(SVRG):
    def __init__(self, game: Game, p: Optional[float] = None) -> None:
        super().__init__(game)
        if p is None:
            p = 1/game.num_samples
        self.p = torch.as_tensor(p)

        self._update_state()

    def update_state(self) -> None:
        if not self.p.bernoulli_():
            self._update_state()


class VRFoRB(LooplessSVRG):
    def __init__(self, game: Game, p: Optional[float] = None) -> None:
        super().__init__(game)
        if p is None:
            p = 1/game.num_samples
        self.p = torch.as_tensor(p)
        self._update_state()

    def _update_state(self) -> None:
        if self.game_copy is None:
            self.game_previous = self.game.copy()
        else:
            self.game_previous = self.game_copy.copy()

        super()._update_state()

    def grad(self, index: int) -> List[torch.Tensor]:
        grad = super().grad(index)
        grad_copy = self.game_previous.operator(index)

        return map(self._grad, grad, grad_copy, self.full_grad)