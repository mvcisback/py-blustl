import magnum
from magnum.mpc import mpc

from nose2.tools import params
import unittest


class TestMPC(unittest.TestCase):
    def test_smoke_mpc(self):
        g = magnum.from_yaml("examples/feasible_example.yaml")
        g = magnum.discretize_game(g)
        predictions = mpc(g)
        next(predictions)
        next(predictions)
