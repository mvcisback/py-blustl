"""
TODO: Move dynamics in STL
TODO: Annotate stl with priority 
TODO: Annotate stl with name
TODO: Annotate stl with changeability
TODO: Game to pdf
"""

from itertools import product, chain, starmap, repeat
from functools import partial
from collections import namedtuple
import operator as op
from math import floor, ceil

import yaml
from funcy import pluck, group_by, drop, walk_values, compose
from lenses import lens

from blustl import stl
from blustl.stl_parser import parse_stl, parse_matrix

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


def game_to_sl(g:Game) -> "SL":
    phi = game_to_stl(g)  
    return stl_to_sl(phi, discretize=partial(discretize, dt=g.dt, N=g.N))
    

def step(t:float, dt:float) -> int:
    return int(t / dt)


def discretize(interval, dt, N):
    f = lambda x: min(step(x, dt=dt), N)
    t_0, t_f = interval
    return range(f(t_0), f(t_f) + 1)


def stl_to_sl(phi:"STL", discretize) -> "SL":
    """Returns STL formula with temporal operators erased"""
    return _stl_to_sl([phi], curr_len=lens()[0], discretize=discretize)[0]
    

def _stl_to_sl(phi, *, curr_len, discretize):
    """Returns STL formula with temporal operators erased"""
    # Warning: _heavily_ uses the lenses library
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


def from_yaml(content:str) -> Game:
    g = yaml.load(content)
    sys = tuple(parse_stl(x) for x in g.get('sys', []))
    env = tuple(parse_stl(x) for x in g.get('env', []))
    init = [parse_stl(x) for x in g['init']]
    phi = Phi(sys, env, init)
    ss = SS(*map(parse_matrix, op.itemgetter('A', 'B')(g['state_space'])))
    width = g.get('explore_width', 5)
    dyn = Dynamics(ss, g['num_vars'], g['num_sys_inputs'], g['num_env_inputs'])
    dt = int(g['dt'])
    tf = g['time_horizon']
    steps = int(ceil(int(tf) / dt))
    
    assert ss.A.shape == (dyn.n_vars, dyn.n_vars)
    assert ss.B.shape == (dyn.n_vars, dyn.n_sys + dyn.n_env)

    return Game(phi, dyn, width, dt, steps, tf)
