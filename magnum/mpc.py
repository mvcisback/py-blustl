"""Module for taking a 1 off game and turning into a MPC games."""
# TODO: incorporate external measurements
from collections import deque

import stl
import sympy as sym
import funcy as fn
from lenses import lens

from magnum.game import discretize_stl, discretize_game
from magnum.game import Game, Specs
from magnum.solvers.milp import encode_and_run
from magnum.solvers.cegis import cegis
from magnum.utils import to_lineq


def mpc_games(g: Game, endless=False) -> [Game]:
    """Transform game into sequence of MPC games."""
    yield g
    spec_lens = lens(g).spec
    H2 = sym.Dummy("H_2")

    def make_mpc_template(phi):
        return stl.utils.param_lens(stl.G(stl.Interval(0, H2), phi))

    def set_horizon(phi_lens, h2):
        return stl.utils.set_params(phi_lens, {H2: h2})

    templates = Specs(*map(make_mpc_template, g.spec))

    for n in range(1, g.model.N):
        spec = Specs(*(set_horizon(pl, n * g.model.dt) for pl in templates))
        g = lens(spec_lens.set(spec)).model.t.set(n)
        yield g

    while endless:
        yield g


def state_to_sl(g: Game, q):
    """Takes measurements and writes appropriate STL.
    Currently assumes piecewise interpolation of measurements.
    """
    def measure_lemma(vals: stl, t: int):
        # TODO: Interval should just be [t, t+g.model.dt)
        # Currently a hack since we don't support open intervals
        psi = stl.G(stl.Interval(t, t + g.model.dt / 2), to_lineq(vals))
        return discretize_stl(psi, g.model)

    return stl.andf(*[measure_lemma(phis, t) for t, phis in
                      enumerate(q) if len(phis) != 0])


def mpc(g: Game, *, endless=True):
    predict = encode_and_run if len(g.model.vars.env) == 0 else cegis
    predict = fn.compose(predict, discretize_game)
    external_meas = set()

    state = deque([], maxlen=g.model.N)
    for g in mpc_games(g, endless):
        g = lens(g).spec.init.set(state_to_sl(g, state))
        prediction = predict(g)
        if not prediction.feasible:
            return prediction
        state.append(prediction.solution.get(g.model.t, dict()))

        # TODO: Incorporate external measurements
        external_meas = yield state[-1], g
        if external_meas is None:
            external_meas = set()
