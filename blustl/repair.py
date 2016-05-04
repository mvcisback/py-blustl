from functools import partial
from itertools import product
from collections import deque

from funcy import mapcat

import stl
from constraint_kinds import UNREPAIRABLE, Kind as K


oo = float('inf')
unbounded = stl.Interval(0, oo)

TEMPORAL_WEAKEN = {
    "G": lambda x: stl.F(stl.G(x.arg, x.interval), unbounded),
    "FG": lambda x: stl.G(stl.F(x.arg, x.interval), unbounded),
    "GF": lambda x: stl.F(x.arg, x.interval),
}

TEMPORAL_STRENGTHEN = {
    "F": lambda x: stl.G(stl.F(x.arg, x.interval), unbounded),
    "GF": lambda x: stl.F(stl.G(x.arg, x.interval), unbounded),
    "FG": lambda x: stl.G(x.arg, x.interval),
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


def _change_structure(phi, n, op):
    return [op(phi, stl.G(
        stl.Pred(i, ineq, oo), unbounded))
            for ineq, i in product(
                ("<=", ">="), range(0, n))]


def weaken_structure(phi, n):
    """phi -> phi or G(x ~ ?)"""
    return _change_structure(phi, n, stl.Or)


def strengthen_structure(phi, n):
    """phi -> phi and G(x ~ ?)"""
    return _change_structure(phi, n, stl.And)


def tune_params(phi):
    raise NotImplementedError


def all_repairs(psi):
    raise NotImplementedError

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
