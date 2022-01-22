import unittest
from gamesopt.train import train, TrainConfig

class TestOptimizer(unittest.TestCase):
    def test_sgda(self):
        config = TrainConfig(num_iter=2)
        train(config)