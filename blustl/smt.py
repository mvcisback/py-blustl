import operator as op
from functools import singledispatch

from pysmt.shortcuts import Symbol, Equals
from pysmt.typing import REAL

import stl

from blustl import game

def sl_to_smt(phi:"SL"):
    store = {(s, t): Symbol(f"{s}[{t}]", REAL) for s, t in game.vars_in_phi(phi)}
    return encode(phi, store)
    
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
    return stl.andf(*(encode(x, s) for x in psi.args))

@encode.register(stl.Or)
def _(psi, s:dict):
    return stl.andf(*(encode(x, s) for x in psi.args))

@encode.register(stl.Neg)
def _(psi, s:dict):
    return ~psi.arg
