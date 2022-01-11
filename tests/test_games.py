import unittest
from gamesopt import games
from gamesopt.games import QuadraticGame
from gamesopt.games import KellyAuction
from gamesopt.games.robust_regression import RobustLinReg


class TestGames(unittest.TestCase):

    def test_quadratic_game(self):
        game = QuadraticGame()
        game.reset()
        index = game.sample()
        game.loss(index)
        game.operator(index)
        game.hamiltonian()
        game.dist2opt()
        game.copy()

    def test_kelly_auction(self):
        game = KellyAuction()
        game.reset()
        game.loss()
        game.operator()
        game.hamiltonian()

    def test_robust(self):
        game = RobustLinReg()
        game.reset()
        game.loss()
        game.operator()
        game.hamiltonian()