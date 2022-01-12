import unittest
from gamesopt.games import QuadraticGame
from gamesopt.optimizer.extragradient import SVRE, EGwithVR
from gamesopt.optimizer.lr import DecreasingLR, FixedLR
from gamesopt.optimizer.sgda import VRAGDA, ProxLSVRGDA, SVRG, ProxSGDA, VRFoRB

class TestOptimizer(unittest.TestCase):

    def test_prox_sgda(self):
        game = QuadraticGame()
        optimizer = ProxSGDA(game)
        optimizer.step()

    def test_prox_l_svrgda(self):
        game = QuadraticGame()
        optimizer = ProxLSVRGDA(game)
        optimizer.step()

    def test_svrg(self):
        game = QuadraticGame()
        optimizer = SVRG(game)
        optimizer.step()

    def test_vrfrob(self):
        game = QuadraticGame()
        optimizer = VRFoRB(game)
        optimizer.step()

    def test_vragda(self):
        game = QuadraticGame()
        optimizer = VRAGDA(game)
        optimizer.step()

    def test_svre(self):
        game = QuadraticGame()
        optimizer = SVRE(game)
        optimizer.step()

    def test_egvr(self):
        game = QuadraticGame()
        optimizer = EGwithVR(game)
        optimizer.step()

    def test_scheduler(self):
        scheduler = FixedLR(1.)
        lr = scheduler(0)
        scheduler = DecreasingLR(100, 1., 1.)
        lr = scheduler(0)