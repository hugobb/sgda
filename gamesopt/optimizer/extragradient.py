from .base import Optimizer, OptimizerOptions
from gamesopt.games import Game
from .lr import LRScheduler
from typing import Optional
import torch
from .prox import Prox


class SVRE(Optimizer):
    def __init__(self, game: Game, options: OptimizerOptions = OptimizerOptions(), prox: Prox = Prox()) -> None:
        super().__init__(game, options, prox)

        if isinstance(self.lr, LRScheduler):
            self.lr = (self.lr,) * game.num_players

        self.p = 1 / game.num_samples
        self.N = torch.tensor([0])
        self.set_state()
        
    def set_state(self) -> None:
        self.game_copy = self.game.copy()
        self.full_grad = self.game_copy.full_operator()
        self.N.geometric_(self.p)
        self.num_grad += self.game.num_samples

    def step(self) -> None:
        index = self.sample()
        game_copy = self.game.copy()
        grad = self.game.operator(index)
        grad_copy = self.game_copy.operator(index)
        update = grad - grad_copy + self.full_grad
        for i in range(self.game.num_players):
            g = self.game.unflatten(i, update)
            self.game.players[i] = self.game.players[i] - self.lr[i](self.k)*g

        self.num_grad += 2*len(index)

        index = self.sample()
        grad = self.game.operator(index)
        grad_copy = self.game_copy.operator(index)
        update = grad - grad_copy + self.full_grad
        for i in range(self.game.num_players):
            g = self.game.unflatten(i, update)
            self.game.players[i] = game_copy.players[i] - self.lr[i](self.k)*g
    
        self.num_grad += 2*len(index)
        self.k += 1
        if (self.k % self.N) == 0:
            self.set_state()


class EGwithVR(Optimizer):
    def __init__(self, game: Game, options: OptimizerOptions = OptimizerOptions(), prox: Prox = Prox()) -> None:
        super().__init__(game, options, prox)
        self.alpha = options.alpha

        self.p = options.p
        if self.p is None:
            self.p = 1/game.num_samples
        self.p = torch.as_tensor(self.p)

        self.set_state()    

    def set_state(self):
        self.game_copy = self.game.copy()
        self.full_grad = self.game.full_operator()
        self.num_grad += self.game.num_samples

    def step(self) -> None:
        mean_players = []
        for i in range(self.game.num_players):
            mean = self.alpha*self.game.players[i] + (1-self.alpha)*(self.game_copy.players[i])
            lr = self.lr(self.k)
            g = self.game.unflatten(i, self.full_grad)
            self.game.players[i] = self.prox(mean - lr*g, lr)
            mean_players.append(mean)

        index = self.sample()
        grad = self.game.operator(index)
        grad_copy = self.game_copy.operator(index)
        update = grad - grad_copy + self.full_grad
        for i  in range(self.game.num_players):
            g = self.game.unflatten(i, update)
            lr = self.lr(self.k)
            self.game.players[i] = self.prox(mean_players[i] - lr*g, lr)

        self.num_grad += 2*len(index)
    
        if torch.bernoulli(self.p):
            self.set_state()

        self.k += 1
