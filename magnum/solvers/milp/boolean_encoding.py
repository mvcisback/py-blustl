import operator as op
from functools import singledispatch

import stl

from magnum.game import Game
from magnum.constraint_kinds import Kind as K, Kind
from magnum.constraint_kinds import Category as C


@singledispatch
def encode(psi, s, t):
    raise NotImplementedError(psi)


@encode.register(stl.LinEq)
def _(psi, s, t):
    x = sum(float(term.coeff) * s[(term.id, t)][0] for term in
            psi.terms)
    # TODO: cleanup
    if psi.op == "=":
        yield x == psi.const, K.PRED_EQ
    elif psi.op == "<":
        yield x <= psi.const, K.PRED_EQ
    elif psi.op == ">":
        yield x >= psi.const, K.PRED_EQ
    elif psi.op == "<=":
        yield x <= psi.const, K.PRED_EQ
    elif psi.op == ">=":
        yield x >= psi.const, K.PRED_EQ


@encode.register(stl.Next)
def _(phi, s, t):
    yield from encode(phi.arg, s, t+1)


@encode.register(stl.And)
def _(phi, s, t):
    for psi in phi.args:
        yield from encode(psi, s, t)


@encode.register(stl.Or)
def _(phi, s, t):
    z_phi = s[phi][0]
    elems = [s[psi][0] for psi in phi.args]

    yield from ((z_phi >= e, K.OR) for e in elems)
    yield sum(elems) >=  z_phi, K.OR_TOTAL

    for psi in phi.args:
        yield from encode(psi, s, t)
