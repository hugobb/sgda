from typing import List, Optional, Tuple
from .base import Game
from dataclasses import dataclass
import torch
from .utils import random_vector
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler


@dataclass
class RobustLinRegConfig:
    num_samples: int = 10
    dim: int = 2
    lambda_coeff: float = 0.2
    gamma_coeff: float = 0.2


class RobustLinReg(Game):
    def __init__(self, config: RobustLinRegConfig = RobustLinRegConfig()) -> None:
        self.dim = config.dim
        players = [torch.zeros(self.dim, requires_grad=True), torch.zeros(self.dim, requires_grad=True)]

        super().__init__(players, config.num_samples)

        self.x, self.y = torch.zeros(self.num_samples, self.dim).normal_(), torch.zeros(self.num_samples).normal_()
        self.lambda_coeff = config.lambda_coeff
        self.gamma_coeff = config.gamma_coeff

    def reset(self) -> None:
        for i in range(self.num_players):
            self.players[i] = random_vector(self.dim)

    def loss(self, index: Optional[int] = None) -> List[torch.Tensor]:
        loss = 0.5*(
            ((self.players[0].view(1, -1)*(self.x[index] + self.players[1].view(1, -1))).sum(-1) - self.y[index])**2 
            + 0.5*self.lambda_coeff*(self.players[0]**2).sum()
            - 0.5*self.gamma_coeff*(self.players[1]**2).sum()
        ).mean()
        return [loss, -loss]

    def sample(self, n: int = 1) -> torch.Tensor:
        return torch.randint(self.num_samples, size=(n, ))

    def sample_batch(self) -> torch.Tensor:
        return torch.arange(self.num_samples).long()


class RobustLogReg(Game):
    def __init__(self) -> None:
        self.dataset = MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        self.dim = 1*28*28
        players = [torch.zeros(self.dim, 10, requires_grad=True), torch.zeros(self.dim, requires_grad=True)]
        self.sampler = None

        super().__init__(players, len(self.dataset))

    def reset(self) -> None:
        for i in range(self.num_players):
            self.players[i] = random_vector(self.dim)
    
    def sample(self, n: int = 1) -> torch.Tensor:
        if self.sampler is None:
            sampler = RandomSampler(self.dataset, replacement=True, num_samples=2 ** 31)
            self.sampler = iter(DataLoader(self.dataset, batch_size=n, shuffle=False,
                      sampler=sampler, num_workers=4, pin_memory=True))
        return next(self.sampler)

    def sample_batch(self) -> Optional[torch.Tensor]:
        return next(iter(DataLoader(self.dataset, batch_size=self.num_samples, shuffle=False, num_workers=4, pin_memory=True)))

    def loss(self, input: Tuple[torch.Tensor, torch.Tensor]) -> List[torch.Tensor]:
        x, y = input
        pred = (x.view(len(x), -1) + self.players[1].view(1, self.dim)).mm(self.players[0])
        loss = F.cross_entropy(pred, y)
        return [loss, -loss]