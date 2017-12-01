from pytest import raises

from magnum.solvers.cegis2 import cegis_loop, MaxRoundsError

def test_counter_examples():
    from magnum.examples.feasible_example2 import feasible_example as g
    
    res, counter_examples = cegis_loop(g)
    assert not res.feasible
    assert len(counter_examples) == 1
    
    with raises(MaxRoundsError):
        cegis_loop(g, max_ce=0)


# TODO
def test_rps():
    from magnum.examples.rock_paper_scissors import rps as g
    
    res, counter_examples = cegis_loop(g, use_smt=True)
    assert not res.feasible
    assert len(counter_examples) == 3

    with raises(MaxRoundsError):
        cegis_loop(g, use_smt=True, max_ce=0)

    # TODO
    res, counter_examples = cegis_loop(g)
    assert not res.feasible
    assert len(counter_examples) == 3

    with raises(MaxRoundsError):
        cegis_loop(g, max_ce=0)
