import operator as op
from operator import itemgetter as ig
from itertools import product
from functools import singledispatch

import funcy as fn
import numpy as np
from bidict import bidict
from pysmt.shortcuts import Symbol, get_model, Plus, And, Or, Equals
from pysmt.operators import LT, LE
from pysmt.typing import REAL, BOOL

import stl

from magnum import game
from magnum.utils import Result


def encode(phi, store=None):
    if store is None:
        store = bidict()

    return _encode(phi, store, t=0), store


GET_OP = {
    "=": lambda x, y: (x <= y) & (x >= y),
    ">=": op.ge,
    "<=": op.le,
    ">": op.gt,
    "<": op.lt,
}


@singledispatch
def _encode(psi, s, t):
    raise NotImplementedError(psi)


@_encode.register(stl.LinEq)
def _encode_lineq(psi, s: bidict, t):
    def _symbolic_term(term):
        if (term, t) not in s:
            s[(term.id, t)] = Symbol(f"{term}[{t}]", REAL)
        return float(term.coeff) * s[(term.id, t)]

    x = Plus([_symbolic_term(term) for term in psi.terms])
    return GET_OP[psi.op](x, psi.const)


@_encode.register(stl.AtomicPred)
def _encode_ap(psi, s: bidict, t):
    if (psi, t) not in s:
        s[(psi.id, t)] = Symbol(f"{psi.id}[{t}]", BOOL)
    return s[(psi.id, t)]


@_encode.register(stl.And)
def _encode_and(psi, s: bidict, t):
    return And(*(_encode(x, s, t) for x in psi.args))


@_encode.register(stl.Or)
def _encode_or(psi, s: bidict, t):
    return Or(*(_encode(x, s, t) for x in psi.args))


@_encode.register(stl.Neg)
def _encode_neg(psi, s: bidict, t):
    return ~_encode(psi.arg, s, t)


@_encode.register(stl.Next)
def _encode_next(psi, s: bidict, t):
    return _encode(psi.arg, s, t + 1)


def decode(smt_expr, s):
    kind = smt_expr.node_type()
    if kind in (LT, LE):
        lhs, rhs = smt_expr.args()
        if lhs.is_constant():
            op = '>' if kind == LT else '>='
            const, terms = lhs, rhs
        else:
            op = '<' if kind == LT else '<='
            const, terms = rhs, lhs

        # Hack to bring back to float
        const = float(str(const))

        # Look up terms in bidict
        _, time = s.inv[terms.args()[0]]

        args = terms.args()
        if not terms.is_plus():
            args = [args]

        terms = tuple(stl.Var(float(str(c)), s.inv[var][0]) for var, c in args)

        lineq = stl.LinEq(terms, op, const)
        return stl.utils.next(lineq, i=time)

    elif smt_expr.is_symbol():
        ap, time = s.inv[smt_expr]
        return stl.utils.next(stl.AtomicPred(ap), i=time)

    children = tuple(decode(arg, s) for arg in smt_expr.args())

    if smt_expr.is_and():
        return stl.And(children)
    elif smt_expr.is_or():
        return stl.Or(children)
    elif smt_expr.is_not():
        return stl.Neg(*children)


def encode_dynamics(g, store=None):
    if store is None:
        store = bidict()

    A, B, C = g.model.dyn
    dt = g.model.dt

    # Adjust for discrete time
    A = np.eye(len(g.model.vars.state)) + dt * A
    B = dt * B
    C = dt * C

    times = list(range(g.model.H + 2))

    for name, t in product(g.model.vars.state, times):
        if (name, t) not in store:
            store[name, t] = Symbol(f"{name}[{t}]", REAL)

    lhses = [
        store[name, t] for name, t in product(g.model.vars.state, times[1:])
    ]

    return And(*(_encode_dynamics(A, B, C, g.model.vars, lhses, store, t)
                 for t in times[:-1])), store


def _encode_dynamics(A, B, C, var_lists, lhs, store, t):
    return And(*(row_to_smt(zip([a, b, c], var_lists), x, store, t)
                 for x, (a, b, c) in zip(lhs, zip(A, B, C))))


def row_to_smt(rows_and_var_lists, lhs, store, t):
    rows_and_var_lists = list(rows_and_var_lists)
    print(rows_and_var_lists)

    def _row_to_smt(rows_and_vars):
        def _create_var(a, x):
            if (x, t) not in store:
                store[(x, t)] = Symbol(f"{x}[{t}]", REAL)

            return float(a) * store[x, t]

        return (_create_var(a, x) for a, x in zip(*rows_and_vars))

    rhs = Plus(fn.mapcat(_row_to_smt, rows_and_var_lists))
    return Equals(lhs, rhs)


def decode_dynmaics(eq, store=None):
    pass


def encode_and_run(g):
    # TODO: add bounds
    phi = game.spec_as_stl(g)
    phi = stl.utils.discretize(phi, g.model.dt)
    f, store = encode(phi)
    f2, store = encode_dynamics(g, store)
    model = get_model(f)
    if model is None:
        return Result(False, None, None)
    solution = fn.group_by(
        ig(0), ((t, s, model[v]) for (s, t), v in store.items()))
    solution = fn.walk_values(
        lambda xs: {k: v.constant_value() for _, k, v in xs}, solution)
    return Result(True, 0, solution)
