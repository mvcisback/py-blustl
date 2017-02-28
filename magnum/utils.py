from collections import namedtuple
import funcy as fn

import stl


def to_lineq(d: dict, t=stl.t_sym):
    return stl.andf(*(stl.LinEq((stl.Var(1, k, t), ), "=", v)
                      for k, v in d.items()))


def project_solution_stl(sol, keys, t):
    vals = [to_lineq(fn.project(v, keys), t=k) for k,v in sol.items() if k >= t]
    return stl.andf(*vals)


Result = namedtuple("Result", ["feasible", "model", "cost", "solution"])
