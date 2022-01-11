import unittest
from gamesopt.optimizer import ProxSGDA, ProxSVRGDA
from gamesopt.games import QuadraticGame
from gamesopt.optimizer.base import Optimizer

class TestOptimizer(unittest.TestCase):

    def test_sgda(self):
        game = QuadraticGame()
        optimizer = ProxSGDA(game)
        index = game.sample()
        optimizer.step(index)

    def test_svrgda(self):
        game = QuadraticGame()
        optimizer = ProxSVRGDA(game)
        index = game.sample()
        optimizer.step(index)
        optimizer.update_state()