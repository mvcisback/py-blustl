import funcy as fn
import stl

import traces
from lenses import bind

from magnum.solvers.smt import encode, decode, encode_and_run

def test_invertability():
    phi = stl.parse('(G[0, 1](x > 4)) | (~(F[0, 1](y < 5)))')
    psi = stl.utils.discretize(phi, 1)
    smt_phi, store = encode(psi)
    psi2 = decode(smt_phi, store)
    assert psi == psi2


def test_feasible_example():
    from magnum.examples.feasible_example import feasible_example as g
    from stl.boolean_eval import pointwise_sat
    res = encode_and_run(g)
    phi = g.spec_as_stl(discretize=False)
    dt = g.model.dt
    assert pointwise_sat(phi, dt=dt)(res.solution)

    res = encode_and_run(g.invert())
    phi = g.spec_as_stl(discretize=False)
    assert not pointwise_sat(phi, dt=dt)(res.solution)


def test_one_player_rps():
    from magnum.examples.rock_paper_scissors import rps as g
    from magnum.examples.rock_paper_scissors import context
    from stl.boolean_eval import pointwise_sat
    res = encode_and_run(g)
    phi = g.spec_as_stl(discretize=False)
    dt = g.model.dt
    assert pointwise_sat(phi, dt=dt)(res.solution)

    not_rock = stl.parse('~(X(xRock))').inline_context(context)
    not_paper = stl.parse('~(X(xPaper))').inline_context(context)
    not_scissors = stl.parse('~(X(xScissors))').inline_context(context)

    g = bind(g).specs.learned.set(not_rock)
    res = encode_and_run(g)
    phi = g.spec_as_stl(discretize=False)
    dt = g.model.dt
    assert pointwise_sat(phi, dt=dt)(res.solution)

    g = bind(g).specs.learned.set(not_rock & not_paper)
    res = encode_and_run(g)
    phi = g.spec_as_stl(discretize=False)
    dt = g.model.dt
    assert pointwise_sat(phi, dt=dt)(res.solution)

    g = bind(g).specs.learned.set(not_rock & not_paper & not_scissors)
    res = encode_and_run(g)
    phi = g.spec_as_stl(discretize=False)
    assert res.solution is None


def test_rps_counter_examples():
    from magnum.examples.rock_paper_scissors import rps as g
    ces = [{'w': traces.TimeSeries([(0, 20/60)])}]

    res = encode_and_run(g, counter_examples=ces)
    assert res.feasible

    ces.append({'w': traces.TimeSeries([(0, 40/60)])})

    res = encode_and_run(g, counter_examples=ces)
    assert res.feasible

    ces.append({'w': traces.TimeSeries([(0, 0)])})

    res = encode_and_run(g, counter_examples=ces)
    assert not res.feasible


    g = g.invert()

    res = encode_and_run(g)
    assert res.feasible
    
    ces = [{'u': traces.TimeSeries([(0, 20/60)])}]

    res = encode_and_run(g, counter_examples=ces)
    assert res.feasible

    ces = [{'u': traces.TimeSeries([(0, 40/60)])}]

    res = encode_and_run(g, counter_examples=ces)
    assert res.feasible

    ces = [({'u': traces.TimeSeries([(0, 0)])})]

    res = encode_and_run(g, counter_examples=ces)
    assert res.feasible
