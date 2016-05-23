# -*- coding: utf-8 -*-

from nose2.tools import params
import unittest

from funcy import pluck

from blustl import milp, stl_parser
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

def test_encode_or():
    # TODO
    pass

def test_encode_and():
    # TODO
    pass

def test_encode_f():
    # TODO
    pass

def test_encode_g():
    # TODO
    pass


def test_encode_neg():
    # TODO
    pass


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

