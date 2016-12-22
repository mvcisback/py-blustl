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
import operator as op
from math import ceil

import yaml
import funcy as fn
from funcy import pluck, group_by, drop, walk_values, compose
import sympy as sym
from lenses import lens

import stl

Specs = namedtuple("Specs", "sys env init dyn")
Game = namedtuple("Game", "spec model meta")
Model = namedtuple("Model", "dt N vars bounds")
Vars = namedtuple("Vars", "state input env")
Meta = namedtuple("Meta", "pri names") # TODO populate


def one_off_game_to_stl(g:Game) -> "STL":
    # TODO: support symbolic matricies
    sys, env = stl.And(g.spec.sys), stl.And(g.spec.env),
    phi = [stl.Or((sys, stl.Neg(env))) if g.spec.env else sys]
    dyn = list(g.spec.dyn)
    init = list(g.spec.init)
    return stl.And(tuple(phi + init + dyn))


def fixed_input_constraint(iden:str):
    terms = (stl.Var(1, i, stl.t_sym),)
    const = sym.Symbol(i + "_star")(stl.t_sym)
    return stl.LinEq(terms, "=", const)


def input_constaints(g:Game) -> "STL":
    inputs = fn.chain(g.model.vars.input, g.model.vars.env)
    return stl.And(tuple(map(fixed_input_constraint, inputs)))


def mpc_game_to_stl(g:Game) -> "STL":
    horizon = stl.Interval(0, g.model.N*g.model.dt)
    prev_horizon = stl.Interval(0, (g.model.N-1)*g.model.dt)
    return stl.And((stl.G(prev_horizon, input_constaints(g)), 
                    stl.G(horizon, phi)))

def negative_time_filter(lineq):
    times = lens(lineq).terms.each_().time.get_all()
    return None if any(t < 0 for t in times) else lineq


filter_none = lambda x: tuple(y for y in x if y is not None)


def discretize_decorator(f):
    @fn.wraps(f)
    def wrapper(g:Game):
        return discretize_stl(f(g), g)
    return wrapper


def discretize_stl(phi:"STL", g:Game) -> "SL":
    phi = game_to_stl(g)  

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


def set_time(*, t, dt=stl.dt_sym, tl=None):
    if tl is None:
        tl = stl.terms_lens(phi)
    focus = tl.tuple_(lens().time, lens().coeff).each_()
    return focus.call("subs", {stl.t_sym: t, stl.dt_sym: dt})


def vars_in_phi(phi):
    focus = stl.terms_lens(phi)
    return set(focus.tuple_(lens().id, lens().time).get_all())


def from_yaml(content:str) -> Game:
    g = defaultdict(list, yaml.load(content))

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
    bounds = g['model']['bounds']
    model = Model(dt=dt, N=steps, vars=Vars(**stl_var_map), bounds=bounds)

    return Game(spec=spec,  model=model, meta=meta)


mpc_game_to_sl = discretize_decorator(mpc_game_to_stl)
one_off_game_to_sl = discretize_decorator(one_off_game_to_stl)
