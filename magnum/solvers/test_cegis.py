import stl
import traces
from pytest import raises
from lenses import bind
from magnum.solvers import smt
import funcy as fn
import numpy as np

from magnum.solvers.cegis import solve, MaxRoundsError, encode_refuted_rec, combined_solver
from magnum.solvers.search_oracle import binary_random_search

def test_counter_examples():
    from magnum.examples.feasible_example2 import feasible_example as g
    
    res, counter_examples = solve(g)
    assert not res.feasible
    assert len(counter_examples) == 1
    
    with raises(MaxRoundsError):
        solve(g, max_ce=0)

def test_binary_search():
    from magnum.examples.rock_paper_scissors import rps as g
    use_smt = True
    g_inv = g.invert()


    solve = smt.encode_and_run if use_smt else combined_solver
    play = solve(g, counter_examples=[])
    solution = fn.project(play.solution, g.model.vars.input)
    counter = solve(g_inv, counter_examples=[solution])
    move = fn.project(counter.solution, g.model.vars.env)

    x0 = np.array([[play.solution[k][0]]
                   for k in g.model.vars.state])

    r_high = binary_random_search(g=g, u_star=solution, w_star=move, x0=x0)





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
