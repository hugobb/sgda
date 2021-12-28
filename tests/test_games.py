import unittest
from gamesopt.games import QuadraticGame
from gamesopt.games import KellyAuction


class TestGames(unittest.TestCase):

    def test_quadratic_game(self):
        game = QuadraticGame()
        game.reset()
        index = game.sample()
        game.loss(index)
        game.operator(index)
        game.hamiltonian()
        game.dist2opt()

    def test_kelly_auction(self):
        game = KellyAuction()
        game.loss()
        game.operator()
        game.hamiltonian()