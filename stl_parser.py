# -*- coding: utf-8 -*-
from parsimonious import Grammar, NodeVisitor
import stl

STL_GRAMMAR = Grammar(u'''
phi = pred / finally / global / neg / or / and / "True"
paren_phi = "(" __ phi __ ")"
or = paren_phi _ "∨" _ paren_phi
and = paren_phi _ "∧" _ paren_phi
finally = "⋄" interval paren_phi
global = "□" interval paren_phi
neg = "¬" paren_phi
id = "x" ~r"\d+"
interval = "[" __ const __ "," __ const __ "]"
pred = id _ op _ const
const = ~r"[\+\-]?\d*(\.\d+)?"
op = ">=" / "<=" / "<" / ">"
_ = ~r"\s"+
__ = ~r"\s"*
''')

class STLVisitor(NodeVisitor):
    def generic_visit(self, _, children): return children
    def visit_phi(self, _, children): return children[0]
    def visit_op(self, op, _): return op.text
    def visit_id(self, name, children): return int(name.text[1:])
    def visit_const(self, const, children): return int(const.text)
    def visit_paren_phi(self, _, (_1, _2, phi, _3, _4)): return phi
    def visit_pred(self, _, (id, _1, op, _3, const)): 
        return stl.Pred(id, op, const)
    def visit_interval(self, _, (_1, _2, left, _3, _4, _5, right, _6, _7)): 
        return stl.Interval(left, right)
    def visit_finally(self, _, (_1, interval, phi)): return stl.F(phi, interval)
    def visit_global(self, _, (_1, interval, phi)): 
        return stl.Neg(stl.F(stl.Neg(phi), interval))
    def visit_neg(self, _, (_1, phi)): return stl.Neg(phi)
    def visit_or(self, _, (phi1, _2, _3, _4, phi2)): return stl.Or(phi1, phi2)
    def visit_and(self, _, (phi1, _2, _3, _4, phi2)): 
        return stl.Neg(stl.Or(stl.Neg(phi1), stl.Neg(phi2)))


def parse_stl(stl_str):
    return STLVisitor().visit(STL_GRAMMAR.parse(stl_str))


def main():
    print(parse_stl("⋄[1,2](x1 < 2)"))
    print(parse_stl("¬(⋄[1,2](x1 < 2))"))
    print(parse_stl("(□[1,3](x1 < 2)) ∨ (x2 >= 4)"))
    print(parse_stl("(□[1,3](x1 < 2)) ∧ (x2 >= 4)"))

if __name__ == '__main__':
    main()
