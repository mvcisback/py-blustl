from functools import singledispatch

import stl
import pyeda.inter
import pyeda.boolalg.expr as pyEdaExpr

def simplify(phi:"STL") -> "STL":
    psi, ap_map = stl.utils.to_mtl(phi)
    simplified_expr = encode_and_simpify(psi)
    return decode(simplified_expr, ap_map)
    

def encode(phi:"MTL"):
    phi_str = str(phi).replace("∨", "|").replace("∧", "&").replace("¬", "~")
    return pyeda.inter.expr(phi_str)

def encode_and_simpify(phi:"MTL"):
    return pyeda.inter.espresso_exprs(encode(phi).to_dnf())[0]

def decode(simplified_expr, ap_map) -> "STL":
    mtl = _decode(simplified_expr)
    return stl.utils.from_mtl(mtl, ap_map)


@singledispatch
def _decode(exp):
    raise NotImplementedError(str(type(exp)))

@_decode.register(pyEdaExpr.Variable)
@_decode.register(pyEdaExpr.Complement)
def _(exp):
    return stl.AtomicPred(str(exp.top))


@_decode.register(pyEdaExpr.AndOp)
def _(exp):
    return stl.And(tuple(_decode(x) for x in exp.xs))


@_decode.register(pyEdaExpr.OrOp)
def _(exp):
    return stl.Or(tuple(_decode(x) for x in exp.xs))
