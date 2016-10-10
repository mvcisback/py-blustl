from nose2.tools import params

from funcy import pluck

from blustl.stl_parser import parse_stl
from blustl import stl_parser, game, stl

@params(
    ('◇[0,1](x1 > 2)', (range(0, 1), range(0, 3))),
    ('□[2,3]◇[0,1](x1 > 2)', (range(0, 1), range(4, 7), range(4, 9))),
    ('(□[2,3]◇[0,1](x1 > 2)) ∧ (◇[0,1](x1 > 2))', 
     (range(1), range(1), range(4, 7), range(4, 9), range(1), range(3))),
)

def test_active_times(x, rngs):
    x = parse_stl(x)
    _rngs = tuple(pluck(1, game.active_times(x, dt=0.5, N=10)))
    assert rngs == _rngs

    @params(*example_ymls)
    def test_from_yaml_smoketest(self, yml_path):
        with open(yml_path) as f:
            from_yaml(f)
