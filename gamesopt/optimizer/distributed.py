from gamesopt.optimizer.prox import Prox
from .base import DistributedOptimizer, OptimizerOptions
from gamesopt.games import Game
import torch.distributed as dist
import torch
from .prox import Prox


class QSGDA(DistributedOptimizer):
    def step(self) -> None:
        index = self.sample()
        grad = self.game.operator(index)
        with torch.no_grad():
            grad, n_bits =self.quantization(grad)
            
            self.n_bits += n_bits
            dist.all_reduce(grad)
            grad /= self.size
            for i in range(self.game.num_players):
                lr = self.lr(self.k)
                g = self.game.unflatten(i, grad) # Reshape the grad to match players shape
                self.game.players[i].data = self.prox(self.game.players[i] - lr*g/self.size, lr)

            self.k += 1
            self.num_grad += len(index)


class DIANA_SGDA(DistributedOptimizer):
    def __init__(self, game: Game, options: OptimizerOptions = OptimizerOptions(), prox: Prox = Prox()) -> None:
        super().__init__(game, options, prox)
        self.alpha = options.alpha
        if self.alpha is None:
            self.alpha = self.quantization.k / self.game.dim

        self.buffer = 0
        self.buffer_server = 0

    def step(self) -> None:
        index = self.sample()
        grad = self.game.operator(index).detach()
        with torch.no_grad():
            delta: torch.Tensor = grad - self.buffer
            delta, n_bits = self.quantization(delta)
            self.buffer = self.buffer + self.alpha*delta

            self.n_bits += n_bits
            dist.all_reduce(delta)
            delta /= self.size
            full_grad = self.buffer_server + delta
            self.buffer_server = self.buffer_server + self.alpha*delta
            for i in range(self.game.num_players):
                lr = self.lr(self.k)
                g = self.game.unflatten(i, full_grad)
                self.game.players[i].data = self.prox(self.game.players[i] - lr*g/self.size, lr)
            
            self.k += 1
            self.num_grad += len(index)

class VR_DIANA_SGDA(DistributedOptimizer):
    def __init__(self, game: Game, options: OptimizerOptions = OptimizerOptions(), prox: Prox = Prox()) -> None:
        super().__init__(game, options, prox)
        self.alpha = options.alpha
        if self.alpha is None:
            self.alpha = self.quantization.k / self.game.dim

        self.p = options.p
        if self.p is None:
            self.p = 1/game.num_samples
        self.p = torch.as_tensor(self.p)

        self.buffer = 0
        self.buffer_server = 0
        
        self.set_state()

    def set_state(self) -> None:
        self.game_copy = self.game.copy()
        self.full_grad = self.game.full_operator().detach()
        self.num_grad += self.game.num_samples

    def step(self) -> None:
        index = self.sample()
        grad = self.game.operator(index).detach()
        grad_copy = self.game.operator(index).detach()

        update = (grad - grad_copy + self.full_grad)

        if torch.bernoulli(self.p):
            self.set_state()

        with torch.no_grad():
            delta: torch.Tensor = update - self.buffer
            delta, n_bits = self.quantization(delta)
            self.buffer = self.buffer + self.alpha*delta

            self.n_bits += n_bits
            dist.all_reduce(delta)
            delta /= self.size
            full_grad = self.buffer_server + delta
            self.buffer_server = self.buffer_server + self.alpha*delta

            for i in range(self.game.num_players):
                lr = self.lr(self.k)
                g = self.game.unflatten(i, full_grad)
                self.game.players[i].data = self.prox(self.game.players[i] - lr*g/self.size, lr)    
            
            self.k += 1
            self.num_grad += 2*len(index)
