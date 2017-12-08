import funcy as fn

from magnum.mpc import echo_env_mpc


def test_mpc_smoke():
    from magnum.examples.feasible_example import feasible_example as g

    results = list(fn.take(10, echo_env_mpc(g)))
    assert results == False
