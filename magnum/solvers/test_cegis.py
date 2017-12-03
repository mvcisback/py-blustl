import stl
import traces
from pytest import raises

from magnum.solvers.cegis import solve, MaxRoundsError, encode_refuted_rec

def test_counter_examples():
    from magnum.examples.feasible_example2 import feasible_example as g
    
    res, counter_examples = solve(g)
    assert not res.feasible
    assert len(counter_examples) == 1
    
    with raises(MaxRoundsError):
        solve(g, max_ce=0)


# TODO
def test_rps():
    from magnum.examples.rock_paper_scissors import rps as g
    
    res, counter_examples = solve(g, use_smt=True)
    assert not res.feasible
    assert len(counter_examples) == 3

    with raises(MaxRoundsError):
        solve(g, use_smt=True, max_ce=0)

    # TODO
    res, counter_examples = solve(g)
    assert not res.feasible
    assert len(counter_examples) == 3

    with raises(MaxRoundsError):
        solve(g, max_ce=0)


def test_encode_refuted_rec():
    refuted = {
        'u1': traces.TimeSeries([(0, 0), (1, 1)]),
        'u2': traces.TimeSeries([(0, 0.5)])
    }
    phi = encode_refuted_rec(refuted, 0.2, [0])
    psi1 = stl.parse('u1 < 0.2')
    psi2 = stl.parse('(u2 > 0.3) | (u2 < 0.7)')
    assert phi == psi1 & psi2

    psi3 = stl.parse('X(u1 > 0.8)')
    psi4 = stl.parse('X((u2 > 0.3) | (u2 < 0.7))')
    phi = encode_refuted_rec(refuted, 0.2, [1])
    assert phi == psi3 & psi4

    phi = encode_refuted_rec(refuted, 0.2, [0, 1])
    assert set(phi.args) == set((psi1 & psi2 & psi3 & psi4).args)
