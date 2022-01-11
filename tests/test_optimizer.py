import unittest
from gamesopt.optimizer import ProxSGDA, ProxSVRGDA
from gamesopt.games import QuadraticGame
from gamesopt.optimizer.lr import DecreasingLR, FixedLR

class TestOptimizer(unittest.TestCase):

    def test_sgda(self):
        game = QuadraticGame()
        optimizer = ProxSGDA(game, FixedLR(1.))
        index = game.sample()
        optimizer.step(index)

    def test_svrgda(self):
        game = QuadraticGame()
        optimizer = ProxSVRGDA(game, FixedLR(1.))
        index = game.sample()
        optimizer.step(index)
        optimizer.update_state()

    def test_scheduler(self):
        scheduler = DecreasingLR(100, 1., 1.)
        lr = scheduler(0)