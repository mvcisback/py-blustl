from collections import deque

import stl

from blustl.game import mpc_games_sl_generator, Game
from blustl.milp import encode_and_run

def queue_to_stl(g:Game, q):
    """Takes measurements and writes appropriate STL.
    Currently assumes piecewise interpolation of measurements.
    TODO: Incorporate Lipshitz bound to bound measurements
    """
    return [stl.G(stl.Interval(t, t+g.model.dt), phi) for t, phi in
            enumerate(q)]


def non_adversarial_mpc_specs(g:Game):
    """Co-routine:
      - Yields: MPC STL
      - Recieves: Set of LinEqs (called measurements)

    TODO: Incorporate Lipshitz bound to bound measurements
    """
    q = deque([], maxlen=g.model.N)
    for phi in mpc_games_sl_generator(g):
        measurements = yield stl.And(tuple([phi] + queue_to_stl(g, q)))
        q.append(measurements)


def non_adversarial_predict(phi, _):
    return encode_and_run(phi)
