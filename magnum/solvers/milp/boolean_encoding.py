from functools import singledispatch, wraps

import stl
from optlang import Constraint, Variable
from magnum.constraint_kinds import Kind as K


eps = 1e-7
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
def z(x: "SL", i: int):
    # TODO: come up with better function name
    cat = 'binary' if isinstance(x[0], stl.LinEq) else 'continuous'
    if isinstance(x[0], stl.LinEq):
        prefix = "q"
    else:
        prefix = "z"
    kwargs = {"name": "{}{}".format(prefix, i)}
    return Variable(type=cat, **kwargs)


@singledispatch
def encode(psi, s, t, within_or=False):
    raise NotImplementedError(psi)


@encode.register(stl.LinEq)
def encode_lineq(psi, s, t, within_or=False):
    x = sum(float(term.coeff) * s[(term.id, t)][0] for term in psi.terms)

    if not within_or:
        if psi.op == "=":
            lb = ub = psi.const
        elif psi.op in ("<", "<="):
            lb, ub = None, psi.const
        elif psi.op in (">", ">="):
            lb, ub = psi.const, None
        yield Constraint(x, lb=lb, ub=ub), psi

    else:
        z_phi = z((psi, t))
        s[psi, t, 'or'] = z_phi
        x = x - psi.const if psi.op in (">", ">=") else psi.const - x
        yield Constraint(x - M * z_phi + eps, ub=0), psi
        yield Constraint(-x - M * (1 - z_phi) + eps, ub=0), psi


@encode.register(stl.Next)
def encode_next(phi, s, t, within_or=False):
    yield from encode(phi.arg, s, t + 1, within_or)
    if within_or:
        s[phi, t, 'or'] = s[phi.arg, t + 1, 'or']


@encode.register(stl.And)
def encode_and(phi, s, t, within_or=False):
    if within_or:
        raise NotImplementedError

    for psi in phi.args:
        yield from encode(psi, s, t, within_or)


@encode.register(stl.Or)
def encode_or(phi, s, t, within_or=False):
    if within_or:
        raise NotImplementedError

    # Shallow encoding of or constraint
    # For at least one of childs to be satisified
    for psi in phi.args:
        yield from encode(psi, s, t, within_or=True)

    elems = [s[psi, t, 'or'] for psi in phi.args]
    yield Constraint(sum(elems), lb=0.5), K.OR_TOTAL
