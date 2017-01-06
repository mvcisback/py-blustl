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
    return stl.andf(*[measure_lemma(phis, t) for t, phis in enumerate(q)])


def updated(meas, q):
    state = {x.terms[0].id: x for x in q[-1]}
    state.update({x.terms[0].id: x for x in meas})
    return set(state.values())
    

def specs(g:Game):
    """Co-routine:
      - Yields: MPC STL
      - Recieves: Set of LinEqs (called measurements)

    TODO: Incorporate Lipshitz bound to bound measurements
    """
    init = set((set_time(t=0, phi=phi) for phi in g.spec.init))
    q = deque([init], maxlen=g.model.N)
    for phi in mpc_games_sl_generator(g):
        meas = yield phi & queue_to_sl(g, q)
        q[-1] = updated(meas, q)
        predicts = meas - q[-1]
        if predicts:
            q.append(predicts)


def predict(phi, g):
    return encode_and_run(phi, g)
