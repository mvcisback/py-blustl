import operator as op
from operator import itemgetter
from functools import singledispatch, reduce

import funcy as fn
from pysmt.shortcuts import Symbol, Equals, get_model
from pysmt.typing import REAL

import stl

from blustl import game

def sl_to_smt(phi:"SL"):
    store = {(s, t): Symbol(f"{s}[{t}]", REAL) for s, t in game.vars_in_phi(phi)}
    return encode(phi, store), store
    
GET_OP = {
    "=": op.eq,
    ">=": op.ge,
    "<=": op.le,
    ">": op.gt,
    "<": op.lt,
}


@singledispatch
def encode(psi, s):
    raise NotImplementedError(psi)

@encode.register(stl.LinEq)
def _(psi, s:dict):
    if psi.op == "=":
        # There's a bug here...need to fix
        raise NotImplementedError
    x = sum(float(term.coeff)*(s[(term.id, term.time)]+0) for term in psi.terms)
    return GET_OP[psi.op](x, psi.const)


@encode.register(stl.And)
def _(psi, s:dict):
    return reduce(op.and_, (encode(x, s) for x in psi.args))

@encode.register(stl.Or)
def _(psi, s:dict):
    return reduce(op.or_, (encode(x, s) for x in psi.args))

@encode.register(stl.Neg)
def _(psi, s:dict):
    return ~encode(psi.arg, s)


def encode_and_run(phi):
    f, store = sl_to_smt(phi)
    model = get_model(f)
    if model is None:
        return Result(False, None, None, None)
    solution = fn.group_by(ig(0), ((t, s, model[v]) for (s, t), v in store.items()))
    solution = fn.walk_values(lambda xs: {k: v for _, k, v in xs}, solution)
    return Result(True, model, None, solution)
