# TODO: incorporate external measurements
from collections import deque
from itertools import chain, repeat

import funcy as fn
import stl

from magnum.game import mpc_games_sl_generator, Game, discretize_stl, set_time
from magnum.game import Game
from magnum.milp import encode_and_run
from magnum.adversarial import cegis
from magnum.utils import to_lineq


def queue_to_sl(g:Game, q):
    """Takes measurements and writes appropriate STL.
    Currently assumes piecewise interpolation of measurements.
    TODO: Incorporate Lipshitz bound to bound measurements
    """ 
    def measure_lemma(vals:stl, t:int):
        # TODO: Interval should just be [t, t+g.model.dt)
        # Currently a hack since we don't support open intervals
        psi = stl.G(stl.Interval(t, t+g.model.dt/2), to_lineq(vals))
        return discretize_stl(psi, g)

    # TODO: Set time based on position in queue
    return stl.andf(*[measure_lemma(phis, t) for t, phis in
                      enumerate(q) if len(phis) != 0])


def specs(g:Game):
    """Co-routine:
      - Yields: MPC SL
      - Recieves: Set of LinEqs (called measurements)

    TODO: Incorporate Lipshitz bound to bound measurements
    """
    init = {phi.terms[0].id: phi.const for phi in g.spec.init}
    spec_gen = mpc_games_sl_generator(g)
    phi = next(spec_gen)
    yield queue_to_sl(g, [init]) & phi

    q = deque([], maxlen=g.model.N)
    for phi in spec_gen:
        t, predicts, meas = yield queue_to_sl(g, q) & phi
        q.append(predicts)
        # TODO: incorporate meas


def mpc(g:Game):
    mpc_specs = specs(g)
    phi = next(mpc_specs)
    external_meas = set()
    predict = non_adversarial.predict if len(g.model.vars.env) == 0 else cegis
    H = 2*g.model.N - 1
    for t in chain(range(H), repeat(H)):
        prediction = predict(phi, g, t)
        if not prediction.feasible:
            return prediction
        predicted_meas = prediction.solution.get(t, dict())
        phi = mpc_specs.send((t, predicted_meas, external_meas))
        external_meas = yield predicted_meas, phi
        if external_meas is None:
            external_meas = set()
