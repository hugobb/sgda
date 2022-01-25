import unittest
from gamesopt.optimizer.base import OptimizerType
from gamesopt.optimizer.quantization import QuantizationOptions, QuantizationType, load_quantization
from gamesopt.train_distributed import train, TrainDistributedConfig
import torch

class TestOptimizer(unittest.TestCase):

    def test_quantization(self):
        q = load_quantization()
        q(torch.ones(10))

    def test_randk(self):
        config = QuantizationOptions(quantization_type=QuantizationType.RANDK)
        q = load_quantization(config)
        q(torch.ones(10))

    def test_normq(self):
        config = QuantizationOptions(quantization_type=QuantizationType.NORM_Q)
        q = load_quantization(config)
        q(torch.ones(10))

    def test_qsgda(self):
        config = TrainDistributedConfig(num_iter=5)
        train(config)

    def test_diana(self):
        config = TrainDistributedConfig(num_iter=5)
        config.optimizer.optimizer_type = OptimizerType.DIANA_SGDA
        train(config)

    def test_diana_vr(self):
        config = TrainDistributedConfig(num_iter=5)
        config.optimizer.optimizer_type = OptimizerType.VR_DIANA_SGDA
        train(config)