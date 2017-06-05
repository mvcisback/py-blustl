from magnum import parse
from magnum import utils
from magnum import game
from magnum import io
import numpy as np

#from nose2.tools import params
import unittest

with open("examples/feasible_example.bin", "rb") as f:
    g = io.load(f)

class TestGame(unittest.TestCase):
    def test_dynamics_lipschitz(self):
        A = np.diag([1,2])
        B = np.diag([3,4])
        N = 4
        L = utils.dynamics_lipschitz(A, B, N)
        self.assertAlmostEqual(4*(1 + 2 + 4 + 8), L)
