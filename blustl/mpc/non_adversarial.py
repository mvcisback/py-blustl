from collections import deque

import stl

from blustl.game import mpc_games_sl_generator, Game, discretize_stl
from blustl.milp import encode_and_run

def queue_to_sl(g:Game, q):
    """Takes measurements and writes appropriate STL.
    Currently assumes piecewise interpolation of measurements.
    TODO: Incorporate Lipshitz bound to bound measurements
    """
    def measure_lemma(phis, t):
        psi = stl.G(stl.Interval(t, t+g.model.dt), stl.andf(*phis))
        return discretize_stl(psi, g)
    return stl.andf(*[measure_lemma(phis, t) for t, phis in enumerate(q)])


def specs(g:Game):
    """Co-routine:
      - Yields: MPC STL
      - Recieves: Set of LinEqs (called measurements)

    TODO: Incorporate Lipshitz bound to bound measurements
    """
    q = deque([], maxlen=g.model.N)
    for phi in mpc_games_sl_generator(g):
        measurements = yield phi & queue_to_sl(g, q)
        q.append(measurements)


def predict(phi, _):
    return encode_and_run(phi)
