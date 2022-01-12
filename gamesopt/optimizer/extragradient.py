from .base import Optimizer, OptimizerOptions
from gamesopt.games import Game
from .lr import LRScheduler
from typing import Optional
import torch


class SVRE(Optimizer):
    def __init__(self, game: Game, options: OptimizerOptions = OptimizerOptions()) -> None:
        super().__init__(game, options)

        if isinstance(self.lr, LRScheduler):
            self.lr = (self.lr,) * game.num_players

        self.p = 1 / game.num_samples
        self.N = torch.tensor([0])
        self.set_state()
        
    def set_state(self) -> None:
        self.game_copy = self.game.copy()
        self.full_grad = self.game_copy.full_operator()
        self.N.geometric_(self.p)

    def step(self) -> None:
        index = self.sample()
        game_copy = self.game.copy()
        grad = self.game.operator(index)
        grad_copy = self.game_copy.operator(index)
        for lr, p, g, g_copy, f_g  in zip(self.lr, self.game.players, grad, grad_copy, self.full_grad):
            p = p - lr(self.k)*(g - g_copy + f_g)

        index = self.sample()
        grad = self.game.operator(index)
        for lr, p, p_copy, g, g_copy, f_g  in zip(self.lr, self.game.players, game_copy.players, grad, grad_copy, self.full_grad):
            p = p_copy - lr(self.k)*(g - g_copy + f_g)
    
        self.k += 1
        if (self.k % self.N) == 0:
            self.set_state()


class EGwithVR(Optimizer):
    def __init__(self, game: Game, options: OptimizerOptions = OptimizerOptions()) -> None:
        super().__init__(game, options)
        self.alpha = options.alpha

        if options.p is None:
            options.p = 1/game.num_samples
        self.p = torch.as_tensor(options.p)

        self.set_state()    

    def set_state(self):
        self.game_copy = self.game.copy()
        self.full_grad = self.game.full_operator()

    def step(self) -> None:
        mean_players = []
        for i, g in enumerate(self.full_grad):
            mean = self.alpha*self.game.players[i] + (1-self.alpha)*(self.game_copy.players[i])
            self.game.players[i] = self.game.prox(mean - self.lr(self.k)*g)
            mean_players.append(mean)

        index = self.sample()
        grad = self.game.operator(index)
        grad_copy = self.game_copy.operator(index)
        for i, (mean_p, g, g_copy, full_g) in enumerate(zip(self.game.players, mean_players, grad, grad_copy, self.full_grad)):
            self.game.players[i] = self.game.prox(mean_p - self.lr(self.k)*(g - g_copy + full_g))
    
        if self.p.bernoulli_():
            self.set_state()

        self.k += 1
