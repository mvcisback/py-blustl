"""
TODO: Game to pdf
TODO: Include meta information in Game
- Annotate stl with priority 
- Annotate stl with name
- Annotate stl with changeability
TODO: add test to make sure phi is hashable after transformation
TODO: create map from SL expr to matching Temporal Logic term after conversion
TODO: reimplement set_time w.o. term lens
"""

from itertools import product, chain, starmap, repeat
from functools import partial
from collections import namedtuple
import operator as op
from math import floor, ceil

import yaml
import funcy as fn
from funcy import pluck, group_by, drop, walk_values, compose
from lenses import lens

import sympy
import stl

Phi = namedtuple("Phi", "sys env init")
Dynamics = namedtuple("Dynamics", "eq n_vars n_sys n_env")
Game = namedtuple("Game", "phi dyn ti meta")
TimeInfo = namedtuple("TimeInfo", "dt N t_f")
Meta = namedtuple("Meta", []) # TODO populate

def game_to_stl(g:Game) -> "STL":
    # TODO: support symbolic matricies
    sys, env = stl.And(g.phi.sys), stl.And(g.phi.env),
    phi = [stl.Or((sys, stl.Neg(env))) if g.phi.env else sys]
    dyn = list(g.dyn.eq)
    init = list(g.phi.init)
    return stl.And(tuple(phi + init + dyn))


def negative_time_filter(lineq):
    times = lens(lineq).terms.each_().time.get_all()
    return None if any(t < 0 for t in times) else lineq


filter_none = lambda x: [y for y in x if y is not None]

def game_to_sl(g:Game) -> "SL":
    phi = game_to_stl(g)  

    # Erase Modal Ops
    psi = stl_to_sl(phi, discretize=partial(discretize, ti=g.ti))

    # Set time
    focus = stl.lineq_lens(psi, bind=False)
    psi = set_time(t=0, dt=g.ti.dt, tl=focus.bind(psi).terms.each_())

    # Type cast time to int (and forget about sympy stuff)
    psi = focus.bind(psi).terms.each_().time.modify(int)

    # Drop terms from time < 0
    psi = focus.bind(psi).modify(negative_time_filter)
    return stl.and_or_lens(psi).args.modify(filter_none)
    
    

def step(t:float, dt:float) -> int:
    return int(t / dt)


def discretize(interval:stl.Interval, ti:TimeInfo):
    f = lambda x: min(step(x, dt=ti.dt), ti.N)
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

        # Discrete time
        times = discretize(psi.interval)

        # Compute terms lens
        terms = stl.terms_lens(psi.arg)

        psi = Op(tuple(terms.time + i for i in times))
        phi = curr_len.set(psi, state=phi)

    # Recurse and update Phi
    if isinstance(psi, stl.NaryOpSTL):
        child_lens = (curr_len.args[i] for i in range(len(psi.children())))
        for l in child_lens:
            phi = _stl_to_sl(phi, curr_len=l, discretize=discretize)

    elif isinstance(psi, stl.Neg):
        phi = _stl_to_sl(phi, curr_len=curr_len.arg, discretize=discretize)
    return phi


def from_yaml(content:str) -> Game:
    g = yaml.load(content)
    sys = tuple(stl.parse(x) for x in g.get('sys', []))
    env = tuple(stl.parse(x) for x in g.get('env', []))
    init = tuple(stl.parse(x) for x in g.get('init', []))
    phi = Phi(sys, env, init)

    eq = tuple(stl.parse(x) for x in g.get('dyn', []))
    dyn = Dynamics(eq, g['num_vars'], g['num_sys_inputs'], g['num_env_inputs'])

    dt = int(g['dt'])
    tf = g['time_horizon']
    steps = int(ceil(int(tf) / dt))
    ti = TimeInfo(dt=dt, t_f=tf, N=steps)
    return Game(phi=phi, dyn=dyn, ti=ti, meta=Meta())


def set_time(*, t, dt=stl.dt_sym, tl=None):
    if tl is None:
        tl = stl.terms_lens(phi)
    focus = tl.tuple_(lens().time, lens().coeff).each_()
    return focus.call("subs", {stl.t_sym: t, stl.dt_sym: dt})


def vars_in_phi(phi):
    focus = stl.terms_lens(phi)
    return set(focus.tuple_(lens().id, lens().time).get_all())