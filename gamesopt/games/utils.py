import torch
import math


def random_vector(dim: int) -> torch.Tensor:
    x = torch.zeros(dim).normal_() / math.sqrt(dim)
    x.requires_grad_()
    return x