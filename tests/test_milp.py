# -*- coding: utf-8 -*-

from nose2.tools import params
import unittest

from blustl import milp, stl_parser


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


@params([
    '⋄[0,1](x1 > 2)',
    '□[2,3]⋄[0,1](x1 > 2)',
    '(□[2,3]⋄[0,1](x1 > 2)) ∧ (⋄[0,1](x1 > 2))'
])
def test_active_times(x):
    pass
