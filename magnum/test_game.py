from magnum import game
from magnum import mpc

from nose2.tools import params
import unittest

g = game.from_yaml("examples/feasible_example.yaml")


class TestGame(unittest.TestCase):

    def test_set_time(self):
        pass

    def test_discretize(self):
        pass

    def test_discretize_stl(self):
        pass

    def test_negative_time_filter(self):
        pass

    def test_smoke_discrete_mpc_games(self):
        specs = list(mpc.discrete_mpc_games(g))
        self.assertEqual(len(specs), 2)

    def test_smoke_mpc_games(self):
        specs = list(mpc.mpc_games(g))
        self.assertEqual(len(specs), 2)

    def test_smoke_discretize_game(self):
        game.discretize_game(g)

    def game_to_stl(self):
        phi = game.game_to_stl(g)
        phi2 = stl.parse(
            "F[0,2](x > 5) & G[0,2](x + -1*x' + dt*5*u = 0) & (x = 0)")
        self.assertEqual(phi, phi2)

    def test_from_yaml(self):
        self.assertEqual(g.model.dt, 1)
        self.assertEqual(g.model.N, 2)
        self.assertEqual(len(g.model.vars.state), 1)
        self.assertEqual(len(g.model.vars.input), 1)
        self.assertEqual(len(g.model.vars.env), 0)
