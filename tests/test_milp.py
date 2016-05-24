# -*- coding: utf-8 -*-

from nose2.tools import params
import unittest

from funcy import pluck

from blustl import milp, stl_parser, game, stl
from blustl.stl_parser import parse_stl


@params(
    ('examples/feasible_example.yaml', True),
    ('examples/feasible2_example.yaml', True),
    ('examples/infeasible_example.yaml', False)
)
def test_feasibility(path, feasibility):
    with open(path) as f:
        params = stl_parser.from_yaml(f)
    res = milp.encode_and_run(params)
    assert res.feasible == feasibility


class TestEncoding(unittest.TestCase):
    def setUp(self):
        dyn = game.Dynamics(None, 3, 2, 1)
        self.g = game.Game(None, dyn, None, dt=0.5, N=2)
        x = {
            (1, 0): 3, (1, 1): 1, (1, 2): 2, (1, 3): 1,
            (3, 0): 4,
        }
        self.s = milp.Store(self.g, x=x.items())

        self.phi3 = parse_stl('x3 < 3')
        self.phi1 = parse_stl('x1 > 2')
        self.phiF = parse_stl('⋄[0,1](x1 > 2)')
        self.phiOr = parse_stl('(x1 > 2) or (x3 < 3)')
        self.phiG = parse_stl('□[0,1](x1 > 2)')
        self.phiAnd = parse_stl('(x1 > 2) and (x3 < 3)')
        self.phiNeg = stl.Neg(self.phi1)
        self.s.z.update({
            (self.phi3, 0): 0,
            (self.phi1, 0): 1, (self.phi1, 1): 0, (self.phi1, 2): 0,
            (self.phiOr, 0): 1,
            (self.phiAnd, 0): 0,
            (self.phiF, 0): 1,
            (self.phiG, 0): 0,
            (self.phiNeg, 0): 0,
        })

    
    def test_encode_or(self):
        constrs = list(milp.encode(self.phiOr, 0, self.s, self.g))
        assert len(constrs) == 3
        assert any(pluck(0, constrs))
        assert not all(pluck(0, constrs))            

    def test_encode_and(self):
        constrs = list(milp.encode(self.phiAnd, 0, self.s, self.g))
        assert len(constrs) == 3
        assert all(pluck(0, constrs))


    def test_encode_f(self):
        constrs = list(milp.encode(self.phiF, 0, self.s, self.g))
        assert len(constrs) == 4
        assert any(pluck(0, constrs))
        assert not all(pluck(0, constrs))


    def test_encode_g(self):
        constrs = list(milp.encode(self.phiG, 0, self.s, self.g))
        assert len(constrs) == 4
        assert all(pluck(0, constrs))


    def test_encode_neg(self):
        constrs = list(milp.encode(self.phiNeg, 0, self.s, self.g))
        assert len(constrs) == 1
        assert constrs[0][0]



@params(
    ('⋄[0,1](x1 > 2)', (range(0, 1), range(0, 3))),
    ('□[2,3]⋄[0,1](x1 > 2)', (range(0, 1), range(4, 7), range(4, 9))),
    ('(□[2,3]⋄[0,1](x1 > 2)) ∧ (⋄[0,1](x1 > 2))', 
     (range(1), range(1), range(4, 7), range(4, 9), range(1), range(3))),
)
def test_active_times(x, rngs):
    x = parse_stl(x)
    _rngs = tuple(pluck(1, milp.active_times(x, dt=0.5, N=10)))
    assert rngs == _rngs

