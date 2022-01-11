from abc import ABC, abstractmethod
import math


class LRScheduler(ABC):
    @abstractmethod
    def __call__(self, k: int) -> float:
        pass

    def __repr__(self):
        return "lr=%.1e" % self.lr

    def __str__(self):
        return "lr=%.1e" % self.lr


class FixedLR(LRScheduler):
    def __init__(self, lr: float):
        self.lr = lr

    def __call__(self, k: int) -> float:
        return self.lr


class DecreasingLR(LRScheduler):
    def __init__(self, K: int, ell_D: float, mu: float):
        self.K = K
        self.ell_D = ell_D
        self.mu = mu
        self.k0 = math.ceil(self.K / 2)

    def __call__(self, k: int) -> float:
        if (self.K <= 2*self.ell_D / self.mu) or (k <= self.k0) :
            return 1 / (2*self.ell_D)
        else:
            return 2 / (4*self.ell_D + self.mu*(k - self.k0))