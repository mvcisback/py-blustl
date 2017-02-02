from collections import namedtuple

import stl

def to_lineq(d:dict):
    return stl.andf(
        *(stl.LinEq((stl.Var(1, k, stl.t_sym),), "=", v) for k, v in d.items()))

def project_solution_stl(solution, keys):
    return to_lineq(fn.project(solution, keys))

Result = namedtuple("Result", ["feasible", "model", "cost", "solution"])
