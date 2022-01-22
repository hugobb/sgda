from enum import Enum
from dataclasses import dataclass
from typing import Tuple

import torch


class QuantizationType(Enum):
    NONE = "none"
    RANDK = "RandK"
    NORM_Q = "Norm-quantization"

@dataclass
class QuantizationOptions:
    quantization_type: QuantizationType = QuantizationType.NONE
    k: int = 5

def load_quantization(options: QuantizationOptions = QuantizationOptions()):
    if options.quantization_type == QuantizationType.NONE:
        return DefaultQuantization()
    elif options.quantization_type == QuantizationType.RANDK:
        return RandKQuantization(options)
    elif options.quantization_type == QuantizationType.NORM_Q:
        return NormQuantization()
    else:
        raise NotImplementedError()

def get_nbits(x: torch.Tensor) -> int:
    if x.dtype == torch.float64:
        return 64
    elif x.dtype == torch.float32:
        return 32
    else:
        raise NotImplementedError()


class DefaultQuantization:
    def __call__(self, x: torch.Tensor) ->  Tuple[torch.Tensor, int]:
        return x, get_nbits(x)*x.numel()

class RandKQuantization(DefaultQuantization):
    def __init__(self, options: QuantizationOptions) -> None:
        self.k = options.k

    def __call__(self, x: torch.Tensor) ->  Tuple[torch.Tensor, int]:
        n = x.numel()
        indices = torch.randperm(n)[n - self.k]
        x[indices] = 0

        return x, get_nbits(x)*self.k

class NormQuantization(DefaultQuantization):
    def __call__(self, x: torch.Tensor) ->  Tuple[torch.Tensor, int]:
        norm = (x**2).sum()
        xi = torch.zeros_like(x).bernoulli_(abs(x) / norm)
        x = norm * x * xi
        return x, get_nbits(x) + 2*xi.count_nonzero()

