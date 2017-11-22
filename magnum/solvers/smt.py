import operator as op
from operator import itemgetter as ig
from functools import singledispatch, reduce

import funcy as fn
from pysmt.shortcuts import Symbol, get_model, ForAll, Exists
from pysmt.typing import REAL, BooleanType

import stl

from magnum import game
from magnum.utils import Result


def sl_to_smt(phi: "SL", g):
    store = {(s, t): Symbol(f"{s}[{t}]", REAL)
             for s, t in stl.utils.vars_in_phi(phi)}
    return encode(phi, store), store


GET_OP = {
    "=": lambda x, y: (x <= y) & (x >= y),
    ">=": op.ge,
    "<=": op.le,
    ">": op.gt,
    "<": op.lt,
}


@singledispatch
def encode(psi, s):
    raise NotImplementedError(psi)


@encode.register(stl.LinEq)
def _(psi, s: dict):
    x = sum(
        float(term.coeff) * (s[(term.id, term.time)] + 0)
        for term in psi.terms)
    return GET_OP[psi.op](x, psi.const)


@encode.register(stl.AtomicPred)
def _(psi, s: dict):
    return Symbol(s[(psi.id, psi.time)], BooleanType)


@encode.register(stl.And)
def _(psi, s: dict):
    return reduce(op.and_, (encode(x, s) for x in psi.args))


@encode.register(stl.Or)
def _(psi, s: dict):
    return reduce(op.or_, (encode(x, s) for x in psi.args))


@encode.register(stl.Neg)
def _(psi, s: dict):
    return ~encode(psi.arg, s)


def encode_and_run(g):
    # TODO: add bounds
    phi = game.game_to_stl(g)
    phi = game.discretize_stl(phi, g.model)
    f, store = sl_to_smt(phi, g)
    model = get_model(f)
    if model is None:
        return Result(False, None, None)
    solution = fn.group_by(
        ig(0), ((t, s, model[v]) for (s, t), v in store.items()))
    solution = fn.walk_values(lambda xs: {k: v.constant_value() for _, k, v in xs}, solution)
    return Result(True, 0, solution)
