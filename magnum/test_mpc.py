import magnum
from magnum.mpc import mpc
from magnum import io

from nose2.tools import params
import unittest

with open("examples/feasible_example.bin", "rb") as f:
    g = io.load(f)

class TestMPC(unittest.TestCase):
    def test_smoke_mpc(self):
        g2 = magnum.discretize_game(g)
        predictions = mpc(g2)
        next(predictions)
        next(predictions)
