from functools import partial
from itertools import product, chain
from collections import deque

from funcy import mapcat

import stl
from constraint_kinds import UNREPAIRABLE, Kind as K


unbounded = stl.Interval('?', '?')
lower_bounded = lambda x: stl.Interval(x.interval.lower, '?')

TEMPORAL_WEAKEN = {
    "G": lambda x: stl.F(lower_bounded(x), stl.G(x.interval, x.arg)),
    "FG": lambda x: stl.G(lower_bounded(x), stl.F(unbounded, x.arg.arg)),
    "GF": lambda x: stl.F(lower_bounded(x), x.arg.arg),
}

TEMPORAL_STRENGTHEN = {
    "F": lambda x: stl.G(unbounded, stl.F(x.interval, x.arg)),
    "GF": lambda x: stl.F(x.interval, stl.G(unbounded, x.arg.arg)),
    "FG": lambda x: stl.G(x.interval, x.arg.arg),
}

TYPE_STR = {stl.F: "F", stl.G: "G"}

def op_type(phi):
    op = TYPE_STR.get(type(phi))
    if isinstance(phi.arg, stl.ModalOp):
       op += TYPE_STR.get(type(phi.arg))
    return op


def temporal_weaken(phi):
    """G -> FG -> GF -> F"""
    return TEMPORAL_WEAKEN.get(op_type(phi), lambda x: x)(phi)


def temporal_strengthen(phi):
    """G <- FG <- GF <- F"""
    return TEMPORAL_STRENGTHEN.get(op_type(phi), lambda x: x)(phi)


def _change_structure(phi, n, op, modal_op):
    f = lambda (rel, i): op([phi, modal_op(unbounded, stl.Pred(i, rel, '?'))])
    return map(f, product(("<=", ">="), range(0, n)))


def weaken_structure(phi, n):
    """phi -> phi or F(x ~ ?)"""
    return _change_structure(phi, n, stl.Or, modal_op=stl.F)


def strengthen_structure(phi, n):
    """phi -> phi and G(x ~ ?)"""
    return _change_structure(phi, n, stl.And, modal_op=stl.G)


def tune_params(phi):
    raise NotImplementedError


def check_consistent(phi, P, N):
    raise NotImplementedError


def all_repairs(psi, n, strengthen=True):
    temporal, structure = (temporal_strengthen, strengthen_structure) if strengthen \
                          else (temporal_weaken, weaken_structure)
    return chain([temporal(psi)], structure(psi, n))


def candidates(iis):
    return mapcat(
        all_repairs, (x for x, k in iis if k not in UNREPAIRABLE))


def repair_oracle(params):
    iis = yield
    q = deque([iis, params]) # iis queue
    while len(q) > 0:
        iis, params = q.pop()

        for c in candidates(iis):
            # TODO: apply candidate repair to params
            yield params
            _iis = yield
            q.appendleft(_iis)
