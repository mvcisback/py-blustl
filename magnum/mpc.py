# TODO: incorporate external measurements
from collections import deque
from itertools import chain, repeat

import funcy as fn
import stl

from magnum.game import discrete_mpc_games, discretize_stl, set_time
from magnum.game import Game
from magnum.milp import encode_and_run
from magnum.adversarial import cegis
from magnum.utils import to_lineq


def queue_to_sl(g: Game, q):
    """Takes measurements and writes appropriate STL.
    Currently assumes piecewise interpolation of measurements.
    TODO: Incorporate Lipshitz bound to bound measurements
    """
    def measure_lemma(vals: stl, t: int):
        # TODO: Interval should just be [t, t+g.model.dt)
        # Currently a hack since we don't support open intervals
        psi = stl.G(stl.Interval(t, t + g.model.dt / 2), to_lineq(vals))
        return discretize_stl(psi, g)

    # TODO: Set time based on position in queue
    return stl.andf(*[measure_lemma(phis, t) for t, phis in
                      enumerate(q) if len(phis) != 0])


def specs(g: Game):
    """Co-routine:
      - Yields: MPC SL
      - Recieves: Set of LinEqs (called measurements)

    TODO: Incorporate Lipshitz bound to bound measurements
    """
    init = {phi.terms[0].id: phi.const for phi in g.spec.init}
    spec_gen = discretize_mpc_games(g, endless=True)

    # Bootstrap MPC loop
    g = next(spec_gen)
    yield queue_to_sl(g, [init]) & game.game_to_stl(g)

    q = deque([], maxlen=g.model.N)
    for phi in spec_gen:
        t, predicts, meas = yield queue_to_sl(g, q) & game.game_to_stl(g)
        q.append(predicts)
        # TODO: incorporate meas


def mpc(g: Game):
    mpc_specs = specs(g)
    predict = encode_and_run if len(g.model.vars.env) == 0 else cegis
    external_meas = set()

    # Start MPC Interaction
    g = next(mpc_specs)
    while True:
        prediction = predict(g)
        if not prediction.feasible:
            return prediction
        predicted_meas = prediction.solution.get(t, dict())
        phi = mpc_specs.send((t, predicted_meas, external_meas))
        external_meas = yield predicted_meas, phi
        if external_meas is None:
            external_meas = set()
