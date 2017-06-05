from nose2.tools import params
import unittest

import magnum
from magnum.solvers import smt
from magnum import io

with open("examples/feasible_example.bin", "rb") as f:
    g = io.load(f)

class TestSMT(unittest.TestCase):

    def test_smoke_smt(self):
        g = magnum.from_yaml("examples/feasible_example.yaml")
        g = magnum.discretize_game(g)
        magnum.game_to_stl(g)

        res1 = smt.encode_and_run(g)
        self.assertTrue(res1.feasible)
