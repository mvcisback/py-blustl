from collections import namedtuple
import funcy as fn
import pandas as pd
import numpy as np

import stl


def to_lineq(d: dict, t=stl.t_sym):
    return stl.andf(*(stl.LinEq((stl.Var(1, k, t), ), "=", v)
                      for k, v in d.items()))


def project_solution_stl(sol, keys, t):
    vals = [
        to_lineq(fn.project(v, keys), t=k) for k, v in sol.items() if k >= t
    ]
    return [v for v in vals if v is not stl.TOP]


Result = namedtuple("Result", ["feasible", "cost", "solution"])


def result_to_traces(res):
    if res.solution is None:
        return None

    data = {
        k: {str(k2): float(v2)
            for k2, v2 in v.items()}
        for k, v in res.solution.items()
    }
    return pd.DataFrame(data=data).T


def _gen_eigs(A, B, N):
    if all(A.shape) and all(B.shape):
        M = B
        for _ in range(N):
            yield max(abs(np.linalg.svd(M, compute_uv=False)))
            M = A @ M
    else:
        yield 0


def dynamics_lipschitz(A, B, N):
    """
    A: State Dynamics matrix
    B: Control matrix
    N: Horizon

    TODO:
    sum eigen values of:
    A, AB, A^2B, A^3B...
    then take 1 norm of the max eigenvalues

    TODO:
    if B is Identity, then only need to compute
    eig(A) and take scalar powers.

    TODO:
    if A,B are positive semi-definite then just product of
    eigenvalues.
    """
    return sum(_gen_eigs(A, B, N))


def pretty_print(g):
    print(f"""
Specification
=============
spec: {g.spec.obj}
init: {g.spec.init}
bounds: {g.spec.bounds}
learned: {g.spec.learned}

MODEL
=====
    dt: {g.model.dt}, H: {g.model.H} t: {g.model.t}

    A: {g.model.dyn.A}
    B: {g.model.dyn.B}
    C: {g.model.dyn.C}

States:
------
{g.model.vars.state}

Inputs:
------
{g.model.vars.input}

Environment Inputs:
------------------
{g.model.vars.env}


Meta:
    drdu: {g.meta.drdu}, drdw: {g.meta.drdw}
""")
