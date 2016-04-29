from functools import partial
from itertools import product

import stl

oo = float('inf')
unbounded = stl.Interval(0, oo)

TEMPORAL_WEAKEN = {
    stl.G: lambda x: stl.FG(x.arg, unbounded , unbounded),
    stl.FG: lambda x: stl.GF(x.arg, unbounded , unbounded),
    stl.GF: lambda x: stl.F(x.arg, unbounded),
}

TEMPORAL_STRENGTHEN = {
    stl.F: lambda x: stl.GF(x.arg, unbounded, unbounded),
    stl.GF: lambda x: stl.FG(x.arg, unbounded , unbounded),
    stl.FG: lambda x: stl.G(x.arg, unbounded),
}

def temporal_weaken(phi):
    """G -> FG -> GF -> F"""
    return TEMPORAL_WEAKEN.get(type(phi), lambda x: x)(phi)


def temporal_strengthen(phi):
    """G <- FG <- GF <- F"""
    return TEMPORAL_STRENGTHEN.get(type(phi), lambda x: x)(phi)


def _change_structure(phi, n, op):
    return [op(phi, stl.G(stl.Pred(i, ineq, oo), unbounded)) for ineq, i
            in product(("<=", ">="), range(0, n))]


def weaken_structure(phi, n):
    """phi -> phi or G(x ~ ?)"""
    return _change_structure(phi, n, stl.Or)


def strengthen_structure(phi, n):
    """phi -> phi and G(x ~ ?)"""
    return _change_structure(phi, n, stl.And)


def tune_params(phi):
    raise NotImplemented
