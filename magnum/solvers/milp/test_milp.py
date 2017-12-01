import stl
import traces

from magnum.solvers.milp import milp


def test_game_to_milp_smoke():
    from magnum.examples.feasible_example import feasible_example as g
    from stl.boolean_eval import pointwise_sat
    res = milp.game_to_milp(g)


def test_feasible():
    from magnum.examples.feasible_example import feasible_example as g
    from stl.boolean_eval import pointwise_sat
    res = milp.encode_and_run(g)
    phi = g.spec_as_stl(discretize=False)
    dt = g.model.dt
    assert pointwise_sat(phi, dt=dt)(res.solution)
    assert res.cost == 5
    
    res = milp.encode_and_run(g.invert())
    phi = g.spec_as_stl(discretize=False)
    dt = g.model.dt
    assert not pointwise_sat(phi, dt=dt)(res.solution)
    assert res.cost == 5

    
def test_one_player_rps_feasibility():
    from magnum.examples.rock_paper_scissors import rps as g
    from stl.boolean_eval import pointwise_sat
    res = milp.encode_and_run(g)
    phi = g.spec_as_stl(discretize=False)
    dt = g.model.dt
    assert pointwise_sat(phi, dt=dt)(res.solution)
    
    assert res.cost == 10


def test_rps_counter_examples():
    from magnum.examples.rock_paper_scissors import rps as g
    ces = [{'w': traces.TimeSeries([(0, 20/60)])}]

    res = milp.encode_and_run(g, counter_examples=ces)
    assert res.feasible

    # TODO

    ces.append({'w': traces.TimeSeries([(0, 40/60)])})
    res = milp.encode_and_run(g, counter_examples=ces)
    assert res.feasible

    ces.append({'w': traces.TimeSeries([(0, 0)])})
    res = milp.encode_and_run(g, counter_examples=ces)
    assert not res.feasible

    g = g.invert()

    res = milp.encode_and_run(g)
    assert res.feasible
    
    ces = [{'u': traces.TimeSeries([(0, 20/60)])}]

    res = milp.encode_and_run(g, counter_examples=ces)
    assert res.feasible

    ces = [{'u': traces.TimeSeries([(0, 40/60)])}]

    res = milp.encode_and_run(g, counter_examples=ces)
    assert res.feasible

    ces = [({'u': traces.TimeSeries([(0, 0)])})]

    res = milp.encode_and_run(g, counter_examples=ces)
    assert res.feasible

    ces = [({'u': traces.TimeSeries([(0, 1)])})]

    res = milp.encode_and_run(g, counter_examples=ces)
    assert res.feasible



def test_counter_examples():
    from magnum.examples.feasible_example2 import feasible_example as g
    
    res = milp.encode_and_run(g)
    assert res.feasible

    ces = [{'w': traces.TimeSeries([(0, 0)])}]
    res = milp.encode_and_run(g, counter_examples=ces)
    assert res.feasible

    ces = [{'w': traces.TimeSeries([(0, 1)])}]
    res = milp.encode_and_run(g, counter_examples=ces)
    assert not res.feasible

    ces = [{'w': traces.TimeSeries([(0, 1)])},
           {'w': traces.TimeSeries([(0, 0)])}]
    res = milp.encode_and_run(g, counter_examples=ces)
    assert not res.feasible
