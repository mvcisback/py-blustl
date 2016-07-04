"""
TODO: Move dynamics in STL
TODO: support multiple modes (hybrid system)
TODO: Encode time in SMT clause
TODO: Annotate stl with priority 
TODO: Annotate stl with name
TODO: Annotate stl with changeability
TODO: Game to pdf
"""

from itertools import product, chain, starmap, repeat
from functools import partial
from collections import namedtuple
import operator as op
from math import floor

from funcy import pluck, group_by, drop, walk_values, compose

from blustl import stl
from lenses import lens


Phi = namedtuple("Phi", ["sys", "env", "init"])
SS = namedtuple("StateSpace", ["A", "B"])
Dynamics = namedtuple("Dynamics", ["ss" , "n_vars", "n_sys", "n_env"])
Game = namedtuple("Game", ["phi", "dyn", "width", "dt", "N", "t_f"])

def game_to_stl(g:Game) -> "STL":
    # TODO: support symbolic matricies
    # TODO: replace x' with x[t-dt]
    # TODO: conjunct phi with dynamics
    sys, env = stl.And(g.phi.sys), stl.And(g.phi.env)
    phi = stl.Or((sys, stl.Neg(env))) if g.phi.env else sys
    return phi


def game_to_sl(g:Game):
    phi = game_to_stl(g)  
    return stl_to_sl(phi, discretize=partial(discretize, dt=g.dt, N=g.N))
    

def step(t:float, dt:float) -> int:
    return int(t / dt)


def active_times(phi, *, g:Game, t_0:float):
    f = lambda x: min(step(x, dt=g.dt), g.N)
    yield phi, range(f(t_0), f(g.t_f) + 1)
    if not isinstance(phi, stl.LinEq):
        lo, hi = phi.interval if isinstance(phi, stl.ModalOp) else (0, 0)
        t_0 += lo
        t_f = g.t_f + hi
        lo2, hi2 = map(f, (t_0, t_f))
        for child in phi.children():
            yield from active_times(child, t_0=t_0, g=g)


def discretize(interval, dt, N):
    f = lambda x: min(step(x, dt=dt), N)
    t_0, t_f = interval
    return range(f(t_0), f(t_f) + 1)


def stl_to_sl(phi:"STL", discretize) -> "SL":
    """Returns STL formula with temporal operators erased"""
    return _stl_to_sl([phi], curr_len=lens()[0], discretize=discretize)[0]

    

def _stl_to_sl(phi, *, curr_len, discretize):
    """Returns STL formula with temporal operators erased"""
    # TODO: support Until
    psi = curr_len.get(state=phi)

    # Base Case
    if isinstance(psi, stl.LinEq):
        return phi

    # Erase Time
    if isinstance(psi, stl.ModalOp):
        Op = stl.And if isinstance(psi, stl.G) else stl.Or
        tl = time_lens(psi.arg)

        # Discrete time
        times = discretize(psi.interval)

        # TODO: set time to t+i
        psi = Op([tl.bind(psi.arg) + i for i in times])
        phi = curr_len.set(psi, state=phi)

    # Recurse and update Phi
    if isinstance(psi, stl.NaryOpSTL):
        child_lens = (curr_len.args[i] for i in range(len(psi.children())))
        for l in child_lens:
            phi = _stl_to_sl(phi, curr_len=l, discretize=discretize)

    elif isinstance(psi, stl.Neg):
        phi = _stl_to_sl(phi, curr_len=curr_len.arg, discretize=discretize)
        
    return phi


def time_lens(phi:"STL") -> lens:
    if isinstance(phi, stl.LinEq):
        return lens().terms.each_().var.time

    if isinstance(phi, stl.NaryOpSTL):
        child_lens = [lens()[i].add_lens(time_lens(c)) for i, c
                      in enumerate(phi.children())]
        return lens().args.tuple_(*child_lens).each_()
    else:
        return lens().arg.add_lens(time_lens(phi.arg))
