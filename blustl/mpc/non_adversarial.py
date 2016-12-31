from collections import deque

from blustl.game import mpc_games_sl_generator, Game

def queue_to_stl(g:Game, q):
    """Takes measurements and writes appropriate STL.
    Currently assumes piecewise interpolation of measurements.
    TODO: Incorporate Lipshitz bound to bound measurements
    """
    return [stl.G(stl.Interval(t, t+g.model.dt)) for t, phi in enumerate(q)]


def mpc_game(g:Game):
    """Co-routine:
      - Yields: MPC STL
      - Recieves: Measurement STL

    TODO: Incorporate Lipshitz bound to bound measurements
    """
    q = deque([], mexlen=g.model.N)
    for phi in mpc_games_sl_generator(g):
        measurements = yield stl.And(tuple([phi] + queue_to_stl(g, q)))
        q.append(measurement)
