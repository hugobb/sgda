import unittest
from gamesopt import games
from gamesopt.games import QuadraticGame
from gamesopt.games import KellyAuction
from gamesopt.games.bilinear import BilinearGame
from gamesopt.games.quadratic_games import QuadraticGameConfig
from gamesopt.games.robust_regression import RobustLinReg


class TestGames(unittest.TestCase):

    def test_quadratic_game(self):
        game = QuadraticGame()
        game.reset()
        index = game.sample()
        game.loss(index)
        game.operator(index)
        game.hamiltonian()
        game.copy()

    def test_importance_sampling(self):
        config = QuadraticGameConfig(importance_sampling=True)
        game = QuadraticGame(config)
        game.reset()
        index = game.sample()
        game.loss(index)
        game.operator(index)
        game.hamiltonian()
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

    def test_bilinear(self):
        game = BilinearGame()
        game.reset()
        index = game.sample()
        game.loss(index)
        game.operator(index)
        game.hamiltonian()
        game.dist2opt()