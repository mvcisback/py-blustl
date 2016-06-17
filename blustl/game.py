"""
TODO: Move dynamics in STL
TODO: support multiple modes (hybrid system)
TODO: Encode time in SMT clause
TODO: Annotate stl with priority 
TODO: Annotate stl with name
TODO: Annotate stl with changeability

"""

from itertools import product, chain, starmap
from collections import namedtuple
import operator as op
from math import ceil

from funcy import pluck, group_by, drop, walk_values, compose

from blustl import stl

Phi = namedtuple("Phi", ["sys", "env", "init"])
SS = namedtuple("StateSpace", ["A", "B"])
Dynamics = namedtuple("Dynamics", ["ss" , "n_vars", "n_sys", "n_env"])
Game = namedtuple("Game", ["phi", "dyn", "width", "dt", "N"])

def game_to_lstl(g:Game):
    sys, env = stl.And(g.phi.sys), stl.And(g.phi.env)
    # TODO: Compute formula for dynamics 
    # G (x' = A x + B u + C w)
    # but, can't talk about "next" states thus
    # A must update next state and move current state to past 
    # and update current state based on current past state
    # A' = [ 0 A2; 0 I ]
    
    return 

def lstl_to_milp():
    pass

def active_times(phi, *, dt:int, N:int, t_0=0, t_f=0):
    f = lambda x: min(step(x, dt=dt), N)
    yield phi, range(f(t_0), f(t_f) + 1)
    if not isinstance(phi, stl.Pred):
        lo, hi = phi.interval if isinstance(phi, stl.ModalOp) else (0, 0)
        t_0 += lo
        t_f += hi
        lo2, hi2 = map(f, (t_0, t_f))
        for child in phi.children():
            yield from active_times(child, dt=dt, N=N, t_0=t_0, t_f=t_f) 


def step(t:float, dt:float):
    return int(ceil(t / dt))


def encode_state_evolution(g:Game):
    inputs = ["u{}".format()]
    state = lambda t: pluck(t, s.x.values())
    dot = lambda x, y: sum(starmap(op.mul, zip(x, y)))
    A, B = g.dyn.ss
    for t in range(g.N):
        for i, (A_i, B_i) in enumerate(zip(A, B)):
            dx = g.dt*(dot(A_i, state(t)) + dot(B_i, inputs(t)))
            constr = s.x[i][t + 1] == s.x[i][t] + dx
            yield constr, K.DYNAMICS
