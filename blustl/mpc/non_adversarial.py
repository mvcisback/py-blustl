from collections import deque

from blustl.game import mpc_games_sl_generator, Game

def measurements():
    #TODO
    pass


def queue_to_stl():
    pass


def mpc_game(g:Game):
    q = deque([], mexlen=g.model.N)
    for m, phi in zip(measurements, mpc_games_sl_generator(g)):
        q.append(m)
        yield stl.And((phi, queue_to_stl(q)))
