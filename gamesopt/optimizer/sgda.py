from .base import Optimizer, OptimizerOptions
from .lr import LRScheduler
from gamesopt.games import Game
import torch


class ProxSGDA(Optimizer):
    def __init__(self, game: Game, options: OptimizerOptions = OptimizerOptions()) -> None:
        super().__init__(game, options)

    def step(self) -> None:
        index = self.sample()
        grad = self.game.operator(index)
        
        for i, g in enumerate(grad):
            self.game.players[i] = self.game.prox(self.game.players[i] - self.lr(self.k)*g)

        self.k += 1


class ProxLSVRGDA(Optimizer):
    def __init__(self, game: Game, options: OptimizerOptions = OptimizerOptions()) -> None:
        super().__init__(game, options)

        if options.p is None:
            options.p = 1/game.num_samples
        self.p = torch.as_tensor(options.p)

        self.set_state()

    def set_state(self) -> None:
        self.game_copy = self.game.copy()
        self.full_grad = self.game.full_operator()

    def step(self) -> None:
        index = self.sample()
        grad = self.game.operator(index)
        grad_copy = self.game_copy.operator(index)
        
        for p, g, g_copy, f_g  in zip(self.game.players, grad, grad_copy, self.full_grad):
            p = self.game.prox(p - self.lr(self.k)*(g - g_copy + f_g))

        if self.p.bernoulli_():
            self.set_state()

        self.k += 1


class VRFoRB(Optimizer):
    def __init__(self, game: Game, options: OptimizerOptions = OptimizerOptions()) -> None:
        super().__init__(game, options)

        if options.p is None:
            options.p = 1/game.num_samples
        self.p = torch.as_tensor(options.p)

        self.game_copy = self.game.copy()
        self.set_state()

    def set_state(self) -> None:
        self.game_previous = self.game_copy.copy()
        self.game_copy = self.game.copy()
        self.full_grad = self.game.full_operator()

    def step(self) -> None:
        index = self.sample()
        grad = self.game.operator(index)
        grad_copy = self.game_previous.operator(index)
        
        for p, g, g_copy, f_g  in zip(self.game.players, grad, grad_copy, self.full_grad):
            p = self.game.prox(p - self.lr(self.k)*(g - g_copy + f_g))

        if self.p.bernoulli_():
            self.set_state()

        self.k += 1


class SVRG(Optimizer):
    def __init__(self, game: Game, options: OptimizerOptions = OptimizerOptions()) -> None:
        super().__init__(game, options)

        if isinstance(self.lr, LRScheduler):
            self.lr = (self.lr,) * game.num_players

        if options.p is None:
            options.p = 1/game.num_samples
        self.p = torch.as_tensor(options.p)
        self.N = options.N

        self.set_state()

    def set_state(self) -> None:
        self.game_copy = self.game.copy()
        self.full_grad = self.game.full_operator()

    def step(self) -> None:
        index = self.sample()
        grad = self.game.operator(index)
        grad_copy = self.game_copy.operator(index)
        
        for lr, p, g, g_copy, f_g  in zip(self.lr, self.game.players, grad, grad_copy, self.full_grad):
            p = self.game.prox(p - lr(self.k)*(g - g_copy + f_g))

        self.k += 1
        if (self.k % self.N) == 0:
            self.set_state()

        

class VRAGDA(Optimizer):
    def __init__(self, game: Game, options: OptimizerOptions = OptimizerOptions()) -> None:
        super().__init__(game, options)

        if isinstance(self.lr, LRScheduler):
            self.lr = (self.lr,) * game.num_players

        self.p = torch.ones(1)

        self.N = options.N
        self.T = options.T

        self.set_state(game)
        self._game = game.copy()
        
    def set_state(self, game: Game):
        self.game_copy = game.copy()
        self.full_grad = game.full_operator()

    def step(self) -> None:
        for i, full_g in enumerate(self.full_grad):
            index = self.sample()
            grad = self._game.operator(index, i)
            grad_copy = self.game_copy.operator(index, i)
            self._game.players[i] = self._game.players[i] - self.lr[i](self.k)*(grad - grad_copy + full_g)

        if torch.bernoulli(self.p / ((self.k % (self.T*self.N)) + 2)):
            self.game = self._game.copy()

        self.k += 1
        if (self.k % self.N) == 0:
            self.set_state(self._game)
        
        if (self.k  % (self.T*self.N)) == 0:
            self.set_state(self.game)
     
        


