import stl
import traces
from pytest import raises, approx
from lenses import bind
from magnum.solvers import smt
import funcy as fn
import numpy as np

from magnum.solvers.cegis import solve, MaxRoundsError, encode_refuted_rec, combined_solver
from magnum.solvers.search_oracle import find_refuted_radius, smt_radius_oracle

def test_counter_examples():
    from magnum.examples.feasible_example2 import feasible_example as g
    
    res, counter_examples = solve(g)
    assert not res.feasible
    assert len(counter_examples) == 1
    
    with raises(MaxRoundsError):
        solve(g, max_ce=0)


def test_smt_radius_oracle():
    from magnum.examples.rock_paper_scissors import rps as g
    play = {'u': traces.TimeSeries([(0, 0)])}
    counter = {'w': traces.TimeSeries([(0, 20/60)])}
    
    x0 = np.zeros(2)
    oracle = smt_radius_oracle(g=g, play=play, counter=counter)
    
    assert not oracle(r=5/60)
    assert not oracle(r=9/60)
    assert oracle(r=11/60)
    assert oracle(r=1)


def test_find_refuted_radius():
    from magnum.examples.rock_paper_scissors import rps as g
    play = {'u': traces.TimeSeries([(0, 0)])}
    counter = {'w': traces.TimeSeries([(0, 20/60)])}
    
    r = find_refuted_radius(g, play, counter)
    assert approx(1 - r) == 0


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


def test_rpss():
    from magnum.examples.rock_paper_scissors_spock import rps as g
    
    res, counter_examples = solve(g)
    assert res.feasible
    assert len(counter_examples) == 3
    assert approx(res.cost) == 0.5


def test_encode_refuted_rec():
    refuted = {
        'u1': traces.TimeSeries([(0, 0), (1, 1)]),
        'u2': traces.TimeSeries([(0, 0.5)])
    }
    phi = encode_refuted_rec(refuted, 0.2, [0])
    psi1 = stl.parse('u1 > 0.2')
    psi2 = stl.parse('(u2 < 0.3) | (u2 > 0.7)')
    assert phi == psi1 | psi2

    psi3 = stl.parse('X(u1 < 0.8)')
    psi4 = stl.parse('(X(u2 < 0.3)) | (X(u2 > 0.7))')
    phi = encode_refuted_rec(refuted, 0.2, [1])
    assert phi == psi3 | psi4

    phi = encode_refuted_rec(refuted, 0.2, [0, 1])
    assert set(phi.args) == set((psi1 | psi2 | psi3 | psi4).args)


def test_encode_refute_rec_sync():
    from magnum.examples.feasible_example2 import feasible_example as g
    from magnum.solvers import smt
    from magnum.solvers import milp
    from stl.boolean_eval import pointwise_sat
    refuted = { 'u': traces.TimeSeries([(0, 0.5)]) }
    phi = encode_refuted_rec(refuted, 0.0001, g.times)

    g = bind(g).specs.learned.set(phi)
    res = smt.encode_and_run(g)
    dt = g.model.dt
    assert pointwise_sat(phi, dt=dt)(res.solution)

    g = bind(g).specs.learned.set(phi)
    res = milp.encode_and_run(g)
    dt = g.model.dt
    assert pointwise_sat(phi, dt=dt)(res.solution)
