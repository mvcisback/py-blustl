# TODO: incorporate external measurements
from collections import deque

import funcy as fn
import stl

from blustl.game import mpc_games_sl_generator, Game, discretize_stl, set_time
from blustl.milp import encode_and_run

def queue_to_sl(g:Game, q):
    """Takes measurements and writes appropriate STL.
    Currently assumes piecewise interpolation of measurements.
    TODO: Incorporate Lipshitz bound to bound measurements
    """
    def measure_lemma(phis, t):
        # TODO: Interval should just be [t, t+g.model.dt)
        # Currently a hack since we don't support open intervals
        psi = stl.G(stl.Interval(t, t+g.model.dt/2), stl.andf(*phis))
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
    init = set((set_time(t=0, phi=phi) for phi in g.spec.init))
    spec_gen = mpc_games_sl_generator(g)
    phi = next(spec_gen)
    yield queue_to_sl(g, [init]) & phi

    q = deque([], maxlen=g.model.N)
    for phi in spec_gen:
        t, predicts, meas = yield queue_to_sl(g, q) & phi
        q.append(predicts)
        # TODO: incorporate meas


def predict(phi, g):
    return encode_and_run(phi, g)
