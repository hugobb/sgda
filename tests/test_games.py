import unittest
from gamesopt.games import QuadraticGame


class TestGames(unittest.TestCase):

    def test_quadratic_game(self):
        game = QuadraticGame()
        game.reset()
        index = game.sample()
        game.loss(index)
        game.operator(index)
        game.hamiltonian()
        game.dist2opt()