import operator as op
from functools import partial, singledispatch, wraps

import stl
import funcy as fn
import numpy as np
import pulp as lp

from magnum.game import Game
from magnum.constraint_kinds import Kind as K, Kind
from magnum.constraint_kinds import Category as C

M = 1000  # TODO

def counter(func):
    i = 0
    @wraps(func)
    def _func(*args, **kwargs):
        nonlocal i
        i += 1
        return func(*args, i=i, **kwargs)

    return _func


@counter
def z(x: "SL", g, i):
    # TODO: come up with better function name    
    kwargs = {"name": f"r{i}"}
    if isinstance(x[0], str) and isinstance(x[1], int):
        kwargs = {'name': f"{x[0]}_{x[1]}"}
        if x[0] in set(g.model.vars.input) | set(g.model.vars.env):
            kwargs.update({'lowBound': 0, 'upBound': 1})

        
    r_var = lp.LpVariable(cat=C.Real.value, **kwargs)

    if not isinstance(x, (stl.And, stl.Or)):
        return (r_var, )

    bool_vars = {
        arg: lp.LpVariable(cat=C.Bool.value, name=f"p{i}_{j}")
        for j, arg in enumerate(x.args)
    }
    return (r_var, tuple(bool_vars.items()))


@singledispatch
def encode(psi, s, t):
    raise NotImplementedError(psi)


@encode.register(stl.Neg)
def _(phi, s, t):
    yield s[phi][0] == -s[phi.arg][0], K.NEG
    yield from encode(phi.arg, s, t)


@encode.register(stl.Next)
def _(phi, s, t):
    yield from encode(phi.arg, s, t+1)


@encode.register(stl.ast._Top)
@encode.register(stl.ast._Bot)
def _(phi, s, t):
    yield from []


@encode.register(stl.LinEq)
def _(psi, s, t):
    x = sum(
        float(term.coeff) * s[(term.id, t)][0] for term in psi.terms)
    y = s[stl.utils.next(psi, t)]
    if psi.op in (">", ">="):
        yield y == x - psi.const, K.PRED_EQ
    elif psi.op in ("<", "<="):
        yield y == psi.const - x, K.PRED_EQ
    else:
        raise NotImplementedError
        


def encode_op(phi: "SL", s, t, *, k: Kind, isor: bool):
    r_var, bool_vars = s[phi]
    bool_vars = dict(bool_vars)
    # At most one of the bool vars is active (chosen var)
    yield sum(bool_vars.values()) == 1, k[1]

    # For each variable comput r and assert rel to psi r
    elems = [s[psi] for psi in phi.args]
    rel = op.ge if isor else op.le
    for psi, e in zip(phi.args, elems):
        if len(e) > 1:
            e = e[0]
        yield rel(r_var, e), k[0]
        yield e - (1 - bool_vars[psi]) * M <= r_var, k[0]
        yield r_var <= e + M * (1 - bool_vars[psi]), k[0]


    for psi in phi.args:
        yield from encode(psi, s, t)


encode.register(stl.Or)(partial(encode_op, k=(K.OR, K.OR_TOTAL), isor=True))
encode.register(stl.And)(partial(
    encode_op, k=(K.AND, K.AND_TOTAL), isor=False))


def encode_dynamics(g, store):
    A, B, C = g.model.dyn
    dt = g.model.dt

    # Adjust for discrete time
    A = np.eye(len(g.model.vars.state)) + dt * A
    B = dt * B
    C = dt * C

    times = g.times
    yield from fn.cat(_encode_dynamics(A, B, C, g.model.vars, store, t) 
                      for t in times[:-1])


def _encode_dynamics(A, B, C, var_lists, store, t):
    rhses = [row_to_smt(zip([a, b, c], var_lists), store, t)
                    for a, b, c in zip(A, B, C)]
    lhses = [store[v, t+1] for v in var_lists[0]]

    yield from ((lhs == rhs, K.PRED_EQ) for lhs, rhs in zip(lhses, rhses))


def row_to_smt(rows_and_var_lists, store, t):
    rows_and_var_lists = list(rows_and_var_lists)

    def _row_to_smt(rows_and_vars):
        def _create_var(a, x):
            return float(a) * store[x, t][0]

        return (_create_var(a, x) for a, x in zip(*rows_and_vars))

    return sum(fn.mapcat(_row_to_smt, rows_and_var_lists))
