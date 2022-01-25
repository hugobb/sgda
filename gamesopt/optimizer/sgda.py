from gamesopt.optimizer.prox import Prox
from .base import Optimizer, OptimizerOptions
from .lr import LRScheduler
from gamesopt.games import Game
import torch


class ProxSGDA(Optimizer):
    def step(self) -> None:
        index = self.sample()
        grad = self.game.operator(index).detach()
        
        for i in range(self.game.num_players):
            lr = self.lr(self.k)
            g = self.game.unflatten(i, grad)
            self.game.players[i] = self.prox(self.game.players[i] - lr*g, lr)

        self.k += 1
        self.num_grad += len(index)


class ProxLSVRGDA(Optimizer):
    def __init__(self, game: Game, options: OptimizerOptions = OptimizerOptions(), prox: Prox = Prox()) -> None:
        super().__init__(game, options, prox)

        self.p = options.p
        if self.p is None:
            self.p = 1/game.num_samples
        self.p = torch.as_tensor(self.p)

        self.num_full_grad = 0

        self.set_state()

    def set_state(self) -> None:
        self.game_copy = self.game.copy()
        self.full_grad = self.game.full_operator().detach()
        self.num_grad += self.game.num_samples
        self.num_full_grad += 1

    def step(self) -> None:
        index = self.sample()
        grad = self.game.operator(index).detach()
        grad_copy = self.game_copy.operator(index).detach()
        update = grad - grad_copy + self.full_grad
        
        for i in range(self.game.num_players):    
            g = self.game.unflatten(i, update)
            lr = self.lr(self.k)
            self.game.players[i] = self.prox(self.game.players[i] - lr*g, lr)

        if torch.bernoulli(self.p):
            self.set_state()

        self.k += 1
        self.num_grad += 2*len(index)


class VRFoRB(Optimizer):
    def __init__(self, game: Game, options: OptimizerOptions = OptimizerOptions(), prox: Prox = Prox()) -> None:
        super().__init__(game, options, prox)

        self.p = options.p
        if self.p is None:
            self.p = 1/game.num_samples
        self.p = torch.as_tensor(self.p)

        self.game_copy = self.game.copy()
        self.set_state()

    def set_state(self) -> None:
        self.game_previous = self.game_copy.copy()
        self.game_copy = self.game.copy()
        self.full_grad = self.game.full_operator().detach()
        self.num_grad += self.game.num_samples

    def step(self) -> None:
        index = self.sample()
        grad = self.game.operator(index).detach()
        grad_copy = self.game_previous.operator(index).detach()
        update = grad - grad_copy + self.full_grad
        
        for i in range(self.game.num_players):
            g = self.game.unflatten(i, update)
            lr = self.lr(self.k)
            self.game.players[i] = self.prox(self.game.players[i] - lr*g, lr)

        if torch.bernoulli(self.p):
            self.set_state()

        self.k += 1
        self.num_grad += 2*len(index)


class SVRG(Optimizer):
    def __init__(self, game: Game, options: OptimizerOptions = OptimizerOptions(), prox: Prox = Prox()) -> None:
        super().__init__(game, options, prox)

        if isinstance(self.lr, LRScheduler):
            self.lr = (self.lr,) * game.num_players

        self.N = options.N
        if self.N is None:
            self.N = game.num_samples
        
        self.set_state()

    def set_state(self) -> None:
        self.game_copy = self.game.copy()
        self.full_grad = self.game.full_operator().detach()
        self.num_grad += self.game.num_samples

    def step(self) -> None:
        index = self.sample()
        grad = self.game.operator(index).detach()
        grad_copy = self.game_copy.operator(index).detach()
        update = grad - grad_copy + self.full_grad
        
        for i in range(self.game.num_players):
            g = self.game.unflatten(i, update)
            lr = self.lr[i](self.k)
            self.game.players[i] = self.prox(self.game.players[i] - lr*g, lr)

        self.num_grad += 2*len(index)
        self.k += 1
        if (self.k % self.N) == 0:
            self.set_state()
        

class VRAGDA(Optimizer):
    def __init__(self, game: Game, options: OptimizerOptions = OptimizerOptions(), prox: Prox = Prox()) -> None:
        super().__init__(game, options, prox)

        if isinstance(self.lr, LRScheduler):
            self.lr = (self.lr,) * game.num_players

        self.p = torch.ones(1)

        self.N = options.N
        if self.N is None:
            self.N = game.num_samples
        self.T = options.T

        self.set_state(game)
        self._game = game.copy()
        
    def set_state(self, game: Game):
        self.game_copy = game.copy()
        self.full_grad = game.full_operator().detach()
        self.num_grad += self.game.num_samples

    def step(self) -> None:
        for i in range(self.game.num_players):
            index = self.sample()
            grad = self._game.operator(index, i).detach()
            grad_copy = self.game_copy.operator(index, i).detach()
            fg = self.game.unflatten(i, self.full_grad)
            self._game.players[i] = self._game.players[i] - self.lr[i](self.k)*(grad - grad_copy + fg)

        if torch.bernoulli(self.p / ((self.k % (self.T*self.N)) + 2)):
            self.game.set_players(self._game.players)

        self.num_grad += 2*len(index)
        self.k += 1
        if (self.k % self.N) == 0:
            self.set_state(self._game)
        
        if (self.k  % (self.T*self.N)) == 0:
            self.set_state(self.game)
     
        


