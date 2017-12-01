from pytest import raises

from magnum.solvers.cegis import solve, MaxRoundsError

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
