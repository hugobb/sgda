import unittest
from gamesopt.optimizer import SGDA
from gamesopt.games import QuadraticGame

class TestOptimizer(unittest.TestCase):

    def test_sgda(self):
        game = QuadraticGame()
        optimizer = SGDA(game)
        index = game.sample()
        optimizer.step(index)