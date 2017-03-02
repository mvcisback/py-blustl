import operator as op
from functools import partial, singledispatch

import stl
import pulp as lp

from magnum.game import Game
from magnum.constraint_kinds import Kind as K, Kind
from magnum.constraint_kinds import Category as C

M = 1000  # TODO


def z(x: "SL", i: int, g: Game):
    # TODO: come up with better function name
    kwargs = {"name": f"r{i}"}
    if str(x[0]) in g.model.bounds:
        lo, hi = g.model.bounds.get(str(x[0]))
        kwargs = {"lowBound": lo, "upBound": hi, "name": f"{x[0]}_{x[1]}"}
    r_var = lp.LpVariable(cat=C.Real.value, **kwargs)

    if not isinstance(x, (stl.And, stl.Or)):
        return (r_var,)

    bool_vars = {
        arg: lp.LpVariable(
            cat=C.Bool.value,
            name=f"p{i}{j}") for j,
        arg in enumerate(
            x.args)}
    return (r_var, bool_vars)


@singledispatch
def encode(psi, s):
    raise NotImplementedError(psi)


@encode.register(stl.Neg)
def _(phi, s: dict):
    yield s[phi][0] == -s[phi.arg][0], K.NEG


@encode.register(stl.LinEq)
def _(psi, s: dict):
    x = sum(float(term.coeff) * s[(term.id, term.time)][0]
            for term in psi.terms)
    if psi.op == "=":
        # TODO: this should either be -M or M
        yield x == psi.const, K.PRED_EQ
    else:
        # TODO: come up w. better value for eps
        yield s[psi] == x, K.PRED_EQ


def encode_op(phi: "SL", s: dict, *, k: Kind, isor: bool):
    r_var, bool_vars = s[phi]
    # At most one of the bool vars is active (chosen var)
    yield sum(bool_vars.values()) == 1, k[1]

    # For each variable comput r and assert rel to psi r
    elems = [s[psi] for psi in phi.args]
    rel = op.ge if isor else op.le
    for psi, e in zip(phi.args, elems):
        yield rel(r_var, e), k[0]
        yield e - (1 - bool_vars[psi]) * M <= r_var, k[0]
        yield r_var <= e + M * (1 - bool_vars[psi]), k[0]


encode.register(stl.Or)(partial(encode_op, k=(K.OR, K.OR_TOTAL), isor=True))
encode.register(stl.And)(
    partial(encode_op, k=(K.AND, K.AND_TOTAL), isor=False))
