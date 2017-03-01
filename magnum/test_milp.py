import unittest

import stl
from nose2.tools import params
import sympy as sym

from magnum import milp


class TestGame(unittest.TestCase):

    def test_encode(self):
        phi1 = stl.parse(
            "((x[0] > 5) or (x[1] > 5)) & (~(((x[0] > 5) & (x[1] > 5))))")

        a, b, c, d, e, f, x0, x1 = sym.symbols("a b c d e f x0 x1")

        s = {
            phi1: a,
            phi1.args[0]: b,
            phi1.args[0].args[0]: e,
            phi1.args[0].args[1]: f,
            phi1.args[1]: c,
            phi1.args[1].arg: d
        }

        top_level_and_sol = {a <= b, a <= c, b + c - 1 <= a}
        left_or_sol = {b >= e, b >= f, e + f >= b}
        right_and_sol = {d <= e, d <= f, e + f - 1 <= d}
        neg_sol = {1 - d == 1}
        pred0_sol = {}

        top_level_and_res = {x[0] for x in milp.encode(phi1, s)}
        self.assertEqual(len(top_level_and_sol), 3)
        self.assertEqual(top_level_and_sol, top_level_and_res)

        left_or_res = {x[0] for x in milp.encode(phi1.args[0], s)}
        self.assertEqual(len(left_or_res), 3)
        self.assertEqual(left_or_res, left_or_sol)

        neg_res = {x[0] for x in milp.encode(phi1.args[1], s)}
        self.assertEqual(len(neg_res), 1)
        self.assertEqual(neg_res, neg_sol)
