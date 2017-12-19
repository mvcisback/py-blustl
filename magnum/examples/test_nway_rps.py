from magnum.examples.n_way_rps import create_game
from magnum.solvers import cegis


def test_nway_rps():
    for k in range(1, 4):
        g = create_game(k)
        res = cegis.solve(g)
        assert not res.feasible
        assert len(res.counter_examples) == k
