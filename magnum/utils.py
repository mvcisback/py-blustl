from collections import namedtuple
import funcy as fn
import pandas as pd
import numpy as np

import stl


def to_lineq(d: dict, t=stl.t_sym):
    return stl.andf(*(stl.LinEq((stl.Var(1, k, t), ), "=", v)
                      for k, v in d.items()))


def project_solution_stl(sol, keys, t):
    vals = [to_lineq(fn.project(v, keys), t=k)
            for k, v in sol.items() if k >= t]
    return stl.andf(*vals)


Result = namedtuple("Result", ["feasible", "cost", "solution"])


def result_to_pandas(res):
    if res.solution is None:
        return None

    data = {k: {str(k2): v2 for k2, v2 in v.items()}
            for k, v in res.solution.items()}
    return pd.DataFrame(data=data).T


def _gen_eigs(A, B, N):
    M = B
    for _ in range(N):
        yield max(abs(np.linalg.eigvals(M)))
        M = A @ M

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
