import stl

from nose2.tools import params
import unittest

import magnum
from magnum.solvers import cegis

class TestAdv(unittest.TestCase):

    def test_zero_sum_game(self):
        g = magnum.from_yaml("examples/feasible_example.yaml")
        g = magnum.discretize_game(g)
        phi = magnum.game_to_stl(g)

        # 1 player must win, 1 player must lose
        res1 = cegis.cegis(g)
        #res2 = cegis.cegis(~phi, g, 0)
        self.assertIsNotNone(res1)
        #self.assertIsNone(res2)
