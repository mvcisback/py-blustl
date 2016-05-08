from nose2.tools import params
import unittest

from blustl import milp, stl_parser


@params(
    ('examples/feasible_example.yaml', True),
    ('examples/infeasible_example.yaml', False)
)
def test_feasibility(path, feasibility):
    with open(path) as f:
        params = stl_parser.from_yaml(f)
    feasible, _ = milp.encode_and_run(params)
    assert feasible == feasibility

