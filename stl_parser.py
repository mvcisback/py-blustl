# -*- coding: utf-8 -*-
from parsimonious import Grammar, NodeVisitor
from funcy import cat, flatten

import stl

GRAMMAR = Grammar(u'''
psi = "--env--" EOL env* "--sys--" EOL sys*
env = phi EOL
sys = phi (_ rank)? EOL

rank = "[" const "]"

phi = pred / f / fg / gf / g / or / and / "True"
paren_phi = "(" __ phi __ ")"
paren_pred = "(" __ pred __ ")"

or = paren_phi _ ("∨" / "or") _ paren_phi
and = paren_phi _ ("∧" / "and") _ paren_phi
f = F interval paren_pred
g = G interval paren_pred
fg = F interval G interval paren_pred
gf = G interval F interval paren_pred

F = "F" / "⋄"
G = "G" / "□"
id = "x" ~r"\d+"
interval = "[" __ const __ "," __ const __ "]"
pred = id _ op _ const
const = ~r"[\+\-]?\d*(\.\d+)?"
op = ">=" / "<=" / "<" / ">"
_ = ~r"\s"+
__ = ~r"\s"*
EOL = "\\n"
''')

class Visitor(NodeVisitor):
    def generic_visit(self, _, children): return children
    def visit_phi(self, _, children): return children[0]
    def visit_env(self, _, (phi, _1)): return phi
    def visit_sys(self, _, (phi, maybe_rank, _2)): 
        maybe_rank = flatten(maybe_rank)
        rank = maybe_rank[0] if len(maybe_rank) > 0 else 0
        return phi

    def visit_psi(self, _, children): return children[2], children[5]
    def visit_op(self, op, _): return op.text
    def visit_id(self, name, children): return int(name.text[1:])
    def visit_const(self, const, children): return float(const.text)
    def visit_rank(self, _, (_1, const, _2)): return const
    def visit_paren_pred(self, _, (_1, _2, pred, _3, _4)): return pred
    def visit_paren_phi(self, _, (_1, _2, phi, _3, _4)): return phi
    def visit_pred(self, _, (id, _1, op, _3, const)): 
        return stl.Pred(id, op, const)
    def visit_interval(self, _, (_1, _2, left, _3, _4, _5, right, _6, _7)): 
        return stl.Interval(left, right)
    def visit_f(self, _, (_1, interval, phi)): return stl.F(phi, interval)
    def visit_g(self, _, (_1, interval, phi)): return stl.G(phi, interval)
    def visit_fg(self, _, (_1, i1, _2, i2, p)):
        return stl.F(stl.G(p, i2), i1)
    def visit_gf(self, _, (_1, i1, _2, i2, p)):
        return stl.G(stl.F(p, i2), i1)
    def visit_or(self, _, (phi1, _2, _3, _4, phi2)): return stl.Or(phi1, phi2)
    def visit_and(self, _, (phi1, _2, _3, _4, phi2)): return stl.And(phi1, phi2)

def parse(stl_str, rule="psi"):
    return Visitor().visit(GRAMMAR[rule].parse(stl_str))

def main():
    with open('example1.stl', 'r') as f:
        print(parse("".join(f.readlines())))

if __name__ == '__main__':
    main()
