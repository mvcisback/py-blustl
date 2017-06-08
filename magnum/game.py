"""
TODO: Game to pdf
TODO: Create script to automatically generate game spec
TODO: Include meta information in Game
- Annotate stl with priority
- Annotate stl with name
- Annotate stl with changeability
TODO: create map from SL expr to matching Temporal Logic term after conversion
TODO: refactor discretization
"""

from functools import partial
from collections import namedtuple
from math import ceil

import funcy as fn
import numpy as np
import sympy as sym
from lenses import lens


import stl
from stl import STL

Specs = namedtuple("Specs", "obj init learned bounds")
Model = namedtuple("Model", "dt H vars t dyn")
Vars = namedtuple("Vars", "state input env")
Meta = namedtuple("Meta", "pri names drdu drdw")

# x' = x + dt*(Ax + Bu + Cw)
Dynamics = namedtuple("Dynamics", "A B C")


# TODO: Make a more convenient constructor
class Game(namedtuple("_Game", "spec model meta")):
    __slots__ = ()

    def __new__(cls, spec, model, meta):
        self = super(cls, Game).__new__(cls, spec, model, meta)
        return self


def game_to_stl(g: Game) -> STL:
    dyn = matrix_to_dyn_stl(g)
    return g.spec.obj & g.spec.learned & g.spec.init & g.spec.bounds & dyn


def invert_game(g):
    # Swap Spec
    g2 = lens(g).spec.obj.set(~g.spec.obj)

    # Swap Inputs
    g2 = lens(g2).model.vars.env.set(g.model.vars.input)
    g2 = lens(g2).model.vars.input.set(g.model.vars.env)

    # Swap Dynamics Bounds
    g2 = lens(g2).meta.drdu.set(g.meta.drdw)
    g2 = lens(g2).meta.drdw.set(g.meta.drdu)
    return g2


def discretize_game(g: Game) -> Game:
    specs = Specs(*(discretize_stl(spec, m=g.model) for spec in g.spec))
    return lens(g).spec.set(specs)


def negative_time_filter(lineq):
    times = lens(lineq).terms.each_().time.get_all()
    return None if any(t < 0 for t in times) else lineq


filter_none = lambda x: tuple(y for y in x if y is not None)


def discretize_stl(phi: STL, m: Model) -> "LRA":
    # Erase Modal Ops
    psi = stl_to_lra(phi, discretize=partial(discretize, m=m))
    # Set time
    focus = stl.lineq_lens(psi, bind=False)
    psi = set_time(t=0, dt=m.dt, tl=focus.bind(psi).terms.each_())

    # Type cast time to int (and forget about sympy stuff)
    psi = focus.bind(psi).terms.each_().time.modify(float)
    psi = focus.bind(psi).terms.each_().coeff.modify(float)

    # Drop terms from time < 0
    psi = focus.bind(psi).modify(negative_time_filter)
    return stl.and_or_lens(psi).args.modify(filter_none)


def discretize(interval: stl.Interval, m: Model):
    t_0, t_f = interval
    while t_0 <= t_f:
        yield t_0
        t_0 += m.dt


def stl_to_lra(phi: STL, discretize) -> "LRA":
    """Returns STL formula with temporal operators erased"""
    return _stl_to_lra([phi], curr_len=lens()[0], discretize=discretize)[0]


def _stl_to_lra(phi, *, curr_len, discretize):
    """Returns STL formula with temporal operators erased"""
    # Warning: _heavily_ uses the lenses library
    # TODO: support Until
    psi = curr_len.get(state=phi)

    # Base Case
    if isinstance(psi, stl.LinEq):
        return phi

    # Erase Time
    if isinstance(psi, stl.ModalOp):
        binop = stl.andf if isinstance(psi, stl.G) else stl.orf

        # Discrete time
        times = discretize(psi.interval)

        # Compute terms lens
        terms = stl.terms_lens(psi.arg)
        psi = binop(*(terms.time + i for i in times))
        phi = curr_len.set(psi, state=phi)

    # Recurse and update Phi
    if isinstance(psi, stl.NaryOpSTL):
        child_lens = (curr_len.args[i] for i in range(len(psi.children())))
        for l in child_lens:
            phi = _stl_to_lra(phi, curr_len=l, discretize=discretize)

    elif isinstance(psi, stl.Neg):
        phi = _stl_to_lra(phi, curr_len=curr_len.arg, discretize=discretize)

    return phi


def set_time(*, t=stl.t_sym, dt=stl.dt_sym, tl=None, phi=None):
    if tl is None:
        tl = stl.terms_lens(phi)
    focus = tl.tuple_(lens().time, lens().coeff).each_()

    def _set_time(x):
        if hasattr(x, "subs"):
            return x.subs({stl.t_sym: t, stl.dt_sym: dt})
        return x

    return focus.modify(_set_time)


def matrix_to_dyn_stl(g):
    def to_terms(row, syms, t=stl.t_sym):
        return [stl.Var(c, s, t) for s, c in zip(syms, row) if c != 0]

    model, (A, B, C) = g.model, g.model.dyn
    A, B, C = model.dt*A + np.eye(len(model.vars.state)), model.dt*B, model.dt*C
    def row_to_stl(i, row):
        a_row, b_row, c_row = row
        terms = to_terms(a_row, model.vars.state)
        terms += to_terms(b_row, model.vars.input)
        terms += to_terms(c_row, model.vars.env)
        terms.append(stl.Var(-1, model.vars.state[i], stl.t_sym + model.dt))
        return stl.LinEq(tuple(terms), "=", 0)
    
    dyn_constrs = (row_to_stl(i, row) for i, row in enumerate(zip(A, B, C)))
    return stl.alw(stl.andf(*dyn_constrs), lo=0, hi=model.H)
