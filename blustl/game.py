"""
TODO: Game to pdf
TODO: Create script to automatically generate game spec
TODO: Include meta information in Game
- Annotate stl with priority 
- Annotate stl with name
- Annotate stl with changeability
TODO: add test to make sure phi is hashable after transformation
TODO: create map from SL expr to matching Temporal Logic term after conversion
"""

from itertools import product, chain, starmap, repeat
from functools import partial
from collections import namedtuple, defaultdict
from math import ceil
import operator as op
import pathlib


import yaml
import funcy as fn
from funcy import pluck, group_by, drop, walk_values, compose
import sympy as sym
from lenses import lens

import stl
from stl import STL

import blustl.simplify_mtl

Specs = namedtuple("Specs", "sys env init dyn")
Game = namedtuple("Game", "spec model meta")
Model = namedtuple("Model", "dt N vars bounds")
Vars = namedtuple("Vars", "state input env")
Meta = namedtuple("Meta", "pri names") # TODO populate


def one_off_game_to_stl(g:Game) -> STL:
    # TODO: support symbolic matricies
    sys, env = stl.andf(*g.spec.sys), stl.andf(*g.spec.env)
    phi = (sys | ~env) if g.spec.env else sys
    dyn = stl.andf(*g.spec.dyn)
    init = stl.andf(*g.spec.init)
    return phi & init & dyn


def one_off_game_to_sl(g:Game) -> STL:
    return discretize_stl(one_off_game_to_stl(g), g)


def mpc_games_stl_generator(g:Game) -> STL:
    psi = one_off_game_to_stl(g)
    yield psi

    H2 = sym.Dummy("H_2")
    param_lens = stl.utils.param_lens(stl.G(stl.Interval(0, H2), psi))
    
    for n in range(1, g.model.N):
        psi = stl.utils.set_params(param_lens, {H2:n*g.model.dt})
        yield psi

    while True:
        yield psi


def mpc_games_sl_generator(g:Game) -> STL:
    for phi, prev in fn.with_prev(mpc_games_stl_generator(g)):
        yield prev if prev == phi else discretize_stl(phi, g)


def negative_time_filter(lineq):
    times = lens(lineq).terms.each_().time.get_all()
    return None if any(t < 0 for t in times) else lineq


filter_none = lambda x: tuple(y for y in x if y is not None)



def discretize_stl(phi:STL, g:Game) -> "SL":
    # Erase Modal Ops
    psi = stl_to_sl(phi, discretize=partial(discretize, m=g.model))
    # Set time
    focus = stl.lineq_lens(psi, bind=False)
    psi = set_time(t=0, dt=g.model.dt, tl=focus.bind(psi).terms.each_())

    # Type cast time to int (and forget about sympy stuff)
    psi = focus.bind(psi).terms.each_().time.modify(int)
    psi = focus.bind(psi).terms.each_().coeff.modify(float)

    # Drop terms from time < 0
    psi = focus.bind(psi).modify(negative_time_filter)
    return stl.and_or_lens(psi).args.modify(filter_none)
    

def step(t:float, dt:float) -> int:
    return int(t / dt)


def discretize(interval:stl.Interval, m:Model):
    f = lambda x: min(step(x, dt=m.dt), m.N)
    t_0, t_f = interval
    return range(f(t_0), f(t_f) + 1)


def stl_to_sl(phi:STL, discretize) -> "SL":
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
        binop = stl.And if isinstance(psi, stl.G) else stl.Or

        # Discrete time
        times = discretize(psi.interval)

        # Compute terms lens
        terms = stl.terms_lens(psi.arg)
        psi = binop(tuple(terms.time + i for i in times))
        phi = curr_len.set(psi, state=phi)

    # Recurse and update Phi
    if isinstance(psi, stl.NaryOpSTL):
        child_lens = (curr_len.args[i] for i in range(len(psi.children())))
        for l in child_lens:
            phi = _stl_to_sl(phi, curr_len=l, discretize=discretize)

    elif isinstance(psi, stl.Neg):
        phi = _stl_to_sl(phi, curr_len=curr_len.arg, discretize=discretize)
    return phi


def set_time(*, t, dt=stl.dt_sym, tl=None):
    if tl is None:
        tl = stl.terms_lens(phi)
    focus = tl.tuple_(lens().time, lens().coeff).each_()

    def _set_time(x):
        if hasattr(x, "subs"):
            return x.subs({stl.t_sym: t, stl.dt_sym: dt})
        return x

    return focus.modify(_set_time)


def vars_in_phi(phi):
    focus = stl.terms_lens(phi)
    return set(focus.tuple_(lens().id, lens().time).get_all())


def from_yaml(path) -> Game:
    if isinstance(path, (str, pathlib.Path)):
        with pathlib.Path(path).open("r") as f:
            g = defaultdict(list, yaml.load(f))
    else:
        g = defaultdict(list, yaml.load(f))

    # Parse Specs and Meta
    spec_types = ["sys", "env", "init", "dyn"]
    spec_map = {k: [] for k in spec_types}
    pri_map = {}
    name_map = {}
    for kind in spec_types:
        for spec in g[kind]:
            p = stl.parse(spec['stl'])
            name_map[p] = spec.get('name')
            pri_map[p] = spec.get('pri')
            spec_map[kind].append(p)
    spec_map = fn.walk_values(tuple, spec_map)
    spec = Specs(**spec_map)
    meta = Meta(pri_map, name_map)

    # Parse Model
    stl_var_map = fn.merge(
        {'input': [], 'state': [], 'env': []}, 
        g['model']['vars']
    )
    dt = int(g['model']['dt'])
    steps = int(ceil(int(g['model']['time_horizon']) / dt))
    bounds = {k: v.split(",") for k,v in g["model"]["bounds"].items()}
    bounds = {k: (float(v[0][1:]), float(v[1][:-1])) for k,v in bounds.items()}
    model = Model(dt=dt, N=steps, vars=Vars(**stl_var_map), bounds=bounds)

    return Game(spec=spec,  model=model, meta=meta)
