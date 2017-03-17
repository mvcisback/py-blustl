#from nose2.tools import params
import unittest

import magnum
from magnum.solvers import cegis


class TestAdv(unittest.TestCase):

    def test_smoke_cegis(self):
        g = magnum.from_yaml("examples/feasible_example.yaml")
        g = magnum.discretize_game(g)
        phi = magnum.game_to_stl(g)

        res1 = cegis.cegis(g)
        self.assertIsNotNone(res1)
