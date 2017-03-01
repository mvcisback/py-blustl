import operator as op
from functools import partial, singledispatch

import stl
import pulp as lp

from magnum.game import Game
from magnum.constraint_kinds import Kind as K, Kind
from magnum.constraint_kinds import Category as C

def z(x: "SL", i: int, g: Game):
    # TODO: come up with better function name
    cat = C.Bool if isinstance(x, stl.LinEq) else C.Real
    if isinstance(x, stl.LinEq):
        prefix = "q"
    else:
        prefix = "z"
    kwargs = {"name": "{}{}".format(prefix, i)}
    if str(x[0]) in g.model.bounds:
        lo, hi = g.model.bounds.get(str(x[0]))
        kwargs = {"lowBound": lo, "upBound": hi, "name": f"{x[0]}_{x[1]}"}
    return lp.LpVariable(cat=cat.value, **kwargs)

@singledispatch
def encode(psi, s):
    raise NotImplementedError(psi)


@encode.register(stl.LinEq)
def _(psi, s: dict):
    x = sum(float(term.coeff) * s[(term.id, term.time)] for term in psi.terms)
    if psi.op == "=":
        yield x == psi.const, K.PRED_EQ
    else:
        z_t = s[psi]

        M = 1000  # TODO
        # TODO: come up w. better value for eps
        eps = 1e-5 if psi.op in (">=", "<=") else 0
        mu = x - psi.const + \
            eps if psi.op in ("<", "<=") else psi.const - x - eps
        yield -mu <= M * z_t, K.PRED_UPPER
        yield mu <= M * (1 - z_t), K.PRED_LOWER


@encode.register(stl.Neg)
def _(phi, s: dict):
    yield s[phi] == 1 - s[phi.arg], K.NEG


def encode_op(phi: "SL", s: dict, *, k: Kind, isor: bool):
    z_phi = s[phi]
    elems = [s[psi] for psi in phi.args]
    rel, const = (op.ge, 0) if isor else (op.le, 1 - len(elems))

    for e in elems:
        yield rel(z_phi, e), k[0]
    yield rel(const + sum(elems), z_phi), k[1]


encode.register(stl.Or)(partial(encode_op, k=(K.OR, K.OR_TOTAL), isor=True))
encode.register(stl.And)(
    partial(encode_op, k=(K.AND, K.AND_TOTAL), isor=False))
