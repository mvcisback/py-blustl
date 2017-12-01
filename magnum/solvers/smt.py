# TODO: Create store abstraction class

import operator as op
from operator import itemgetter as ig
from itertools import product
from functools import singledispatch

import funcy as fn
import numpy as np
import traces
from lenses import bind
from bidict import bidict
from pysmt.shortcuts import Symbol, get_model, Plus, And, Or, Equals
from pysmt.shortcuts import FALSE, TRUE, FreshSymbol
from pysmt.operators import LT, LE
from pysmt.typing import REAL, BOOL

import stl

from magnum import game
from magnum.utils import Result


def encode(phi, store=None):
    if store is None:
        store = dict()

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
def _encode_lineq(psi, s: dict, t):
    def _symbolic_term(term):
        if (term.id, t) not in s:
            s[(term.id, t)] = Symbol(f"{term}[{t}]", REAL)
        return float(term.coeff) * s[(term.id, t)]

    x = Plus([_symbolic_term(term) for term in psi.terms])
    eq = GET_OP[psi.op](x, psi.const)
    if eq is False:
        return FALSE()
    elif eq is True:
        return TRUE()
    return eq


@_encode.register(stl.AtomicPred)
def _encode_ap(psi, s: dict, t):
    if (psi, t) not in s:
        s[(psi.id, t)] = Symbol(f"{psi.id}[{t}]", BOOL)

    eq = s[(psi.id, t)]

    if eq is False:
        # TODO: hack to encode false
        return FALSE()
    elif eq is True:
        # TODO: hack to encode true
        return TRUE()

    return eq


@_encode.register(stl.And)
def _encode_and(psi, s: dict, t):
    return And(*(_encode(x, s, t) for x in psi.args))


@_encode.register(stl.Or)
def _encode_or(psi, s: dict, t):
    return Or(*(_encode(x, s, t) for x in psi.args))


@_encode.register(stl.Neg)
def _encode_neg(psi, s: dict, t):
    return ~_encode(psi.arg, s, t)


@_encode.register(stl.Next)
def _encode_next(psi, s: dict, t):
    return _encode(psi.arg, s, t + 1)


def decode(smt_expr, s):
    if isinstance(s, dict):
        s = bidict(s)

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
        store = dict()

    A, B, C = g.model.dyn
    dt = g.model.dt

    # Adjust for discrete time
    A = np.eye(len(g.model.vars.state)) + dt * A
    B = dt * B
    C = dt * C

    times = g.times

    for name, t in product(g.model.vars.state, times):
        if (name, t) not in store:
            store[name, t] = Symbol(f"{name}[{t}]", REAL)

    return And(*(_encode_dynamics(A, B, C, g.model.vars, store, t)
                 for t in times[:-1])), store


def _encode_dynamics(A, B, C, var_lists, store, t):
    rhses = [row_to_smt(zip([a, b, c], var_lists), store, t)
                    for a, b, c in zip(A, B, C)]
    lhses = [store[v, t+1] for v in var_lists[0]]
    return And(*(Equals(lhs, rhs) for lhs, rhs in zip(lhses, rhses)))


def row_to_smt(rows_and_var_lists, store, t):
    rows_and_var_lists = list(rows_and_var_lists)

    def _row_to_smt(rows_and_vars):
        def _create_var(a, x):
            if (x, t) not in store:
                store[(x, t)] = Symbol(f"{x}[{t}]", REAL)
            
            return float(a) * store[x, t]

        return (_create_var(a, x) for a, x in zip(*rows_and_vars))

    return sum(fn.mapcat(_row_to_smt, rows_and_var_lists))


def decode_dynamics(eq, store=None):
    pass


def counter_example_store(g, ce):
    dt = g.model.dt
    return {(name, t): trace[dt*t]
            for (name, trace), t in product(ce.items(), g.times)}


def counter_example_subs(g, store, ce):
    names = set(g.model.vars.state)
    return {store[key]: FreshSymbol(REAL) for key in product(names, g.times)}


def extract_ts(name, model, g, store):
    dt = g.model.dt
    # TODO: hack to have to eval this
    # TODO: support extracting H=0 timeseries
    ts = traces.TimeSeries(((dt*t, eval(str(model[store[name, t]]))) 
                             for t in g.times 
                            if (name, t) in store and store[name, t] in model),
                           domain=(0, g.model.H))
    ts.compact()
    return ts


def encode_and_run(g, counter_examples=None):
    # TODO: add bounds
    phi = g.spec_as_stl()

    if counter_examples is None:
        counter_examples = [{}]

    f = TRUE()
    store = {}
    for i, ce in enumerate(counter_examples):
        # Need to inline counter example .
        store.update(counter_example_store(g, ce))
        f1, store = encode(phi, store)
        f2, store = encode_dynamics(g, store)
        f3 = f1 & f2

        # Create new variable names to avoid conflicts.
        subs = counter_example_subs(g, store, ce) if i > 0 else {}
        f &= (f1 & f2).substitute(subs)

    model = get_model(f)


    if model is None:
        return Result(False, None, None)

    sol = {v: extract_ts(v, model, g, store) for v in fn.cat(g.model.vars)}
    return Result(True, 0, sol)
