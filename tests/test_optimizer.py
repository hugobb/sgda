import unittest
from gamesopt.optimizer import ProxSGDA
from gamesopt.games import QuadraticGame
from gamesopt.optimizer.base import OptimizerOptions
from gamesopt.optimizer.extragradient import EGwithVR, Extragradient
from gamesopt.optimizer.lr import DecreasingLR, FixedLR
from gamesopt.optimizer.vr import GradientUpdate, SVRG, LooplessSVRG, UpdateType, VRFoRB

class TestOptimizer(unittest.TestCase):

    def test_gradient_update(self):
        game = QuadraticGame()
        update = GradientUpdate(game)
        index = game.sample()
        update.grad(index)

    def test_svrg(self):
        game = QuadraticGame()
        update = SVRG(game)
        index = game.sample()
        update.grad(index)
        update.update_state()

    def test_loopless_svrg(self):
        game = QuadraticGame()
        update = LooplessSVRG(game)
        index = game.sample()
        update.grad(index)
        update.update_state()

    def test_vr_forb(self):
        game = QuadraticGame()
        update = VRFoRB(game)
        index = game.sample()
        update.grad(index)
        update.update_state()

    def test_sgda(self):
        game = QuadraticGame()
        optimizer = ProxSGDA(game)
        index = game.sample()
        optimizer.step(index)

    def test_extragradient(self):
        game = QuadraticGame()
        optimizer = Extragradient(game)
        index = game.sample()
        optimizer.step(index)

    def test_eg_with_vr(self):
        game = QuadraticGame()
        options = OptimizerOptions()
        options.update_scheme_options.update_scheme = UpdateType.L_SVRG
        optimizer = EGwithVR(game, options)
        index = game.sample()
        optimizer.step(index)

    def test_scheduler(self):
        scheduler = FixedLR(1.)
        lr = scheduler(0)
        scheduler = DecreasingLR(100, 1., 1.)
        lr = scheduler(0)