#from nose2.tools import params
import unittest

import magnum
from magnum.solvers import cegis
from magnum import io

with open("examples/feasible_example.bin", "rb") as f:
    g = io.load(f)

class TestAdv(unittest.TestCase):

    def test_smoke_cegis(self):
        g2 = magnum.discretize_game(g)
        phi = magnum.game_to_stl(g2)

        res1 = cegis.cegis(g2)
        self.assertIsNotNone(res1)
