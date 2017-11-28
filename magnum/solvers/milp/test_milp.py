import stl

from magnum.solvers.milp import milp


def test_create_store():
    obj = stl.parse('(x < 3) & (y < 3)')
    store = milp.create_store(obj, set(), [0, 1, 2])
    assert len(store) == 2*3 + 3


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
    
    res = milp.encode_and_run(g.invert())
    phi = g.spec_as_stl(discretize=False)
    dt = g.model.dt
    assert not pointwise_sat(phi, dt=dt)(res.solution)
    
