import stl

def to_lineq(d:dict):
    return stl.andf(
        *(stl.LinEq((stl.Var(1, k, stl.t_sym),), "=", v) for k, v in d.items()))
