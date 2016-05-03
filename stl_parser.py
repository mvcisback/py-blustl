# -*- coding: utf-8 -*-
from parsimonious import Grammar, NodeVisitor
from funcy import cat, flatten
import yaml
import numpy as np

import stl


# TODO: allow parsing multiple ors & ands together
STL_GRAMMAR = Grammar(u'''
env = phi
sys = phi (_ rank)?

rank = "[" const "]"

phi = pred / f / fg / gf / g / or / and
phi2 = pred / or2 / and2
pred = id _ op _ const

paren_phi = "(" __ phi __ ")"
paren_phi2 = "(" __ phi2 __ ")"

or = paren_phi _ ("∨" / "or") _ paren_phi
and = paren_phi _ ("∧" / "and") _ paren_phi
or2 = paren_phi2 _ ("∨" / "or") _ paren_phi2
and2 = paren_phi2 _ ("∧" / "and") _ paren_phi2

f = F interval paren_phi2
g = G interval paren_phi2
fg = F interval G interval paren_phi2
gf = G interval F interval paren_phi2

F = "F" / "⋄"
G = "G" / "□"
interval = "[" __ const __ "," __ const __ "]"

id = "x" ~r"\d+"
const = ~r"[\+\-]?\d*(\.\d+)?"
op = ">=" / "<=" / "<" / ">" / "="
_ = ~r"\s"+
__ = ~r"\s"*
EOL = "\\n"
''')

MATRIX_GRAMMAR = Grammar(r'''
matrix = "[" __ row+ __ "]"
row = consts ";"? __
consts = (const _ consts) / const

const = ~r"[\+\-]?\d+(\.\d+)?"
_ = ~r"\s"+
__ = ~r"\s"*
'''
)

class STLVisitor(NodeVisitor):
    def generic_visit(self, _, children): return children

    def visit_env(self, _, phi): return phi
    def visit_sys(self, _, (phi, maybe_rank)): 
        maybe_rank = flatten(maybe_rank)
        rank = maybe_rank[0] if len(maybe_rank) > 0 else 0
        return phi
    def visit_rank(self, _, (_1, const, _2)): return const

    def visit_phi(self, _, children): return children[0]
    def visit_phi2(self, _, children): return children[0]

    def visit_paren_phi(self, _, (_1, _2, phi, _3, _4)): return phi
    def visit_paren_phi2(self, _, (_1, _2, phi, _3, _4)): return phi
    def visit_pred(self, _, (id, _1, op, _3, const)): 
        return stl.Pred(id, op, const)

    def visit_interval(self, _, (_1, _2, left, _3, _4, _5, right, _6, _7)): 
        return stl.Interval(left, right)

    def visit_f(self, _, (_1, interval, phi)): return stl.F(phi, interval)
    def visit_g(self, _, (_1, interval, phi)): return stl.G(phi, interval)
    def visit_fg(self, _, (_1, i1, _2, i2, p)): return stl.F(stl.G(p, i2), i2)
    def visit_gf(self, _, (_1, i1, _2, i2, p)): return stl.G(stl.F(p, i2), i1)

    def visit_or(self, _, (phi1, _2, _3, _4, phi2)): return stl.Or(phi1, phi2)
    def visit_or2(self, _, (phi1, _2, _3, _4, phi2)): return stl.Or(phi1, phi2)
    def visit_and(self, _, (phi1, _2, _3, _4, phi2)): return stl.And(phi1, phi2)
    def visit_and2(self, _, (phi1, _2, _3, _4, phi2)): return stl.And(phi1, phi2)

    def visit_op(self, op, _): return op.text
    def visit_id(self, name, children): return int(name.text[1:])
    def visit_const(self, const, children): return float(const.text)

class MatrixVisitor(NodeVisitor):
    def generic_visit(self, _, children): return children
    def visit_matrix(self, _, (_1, _2, rows, _3, _4)): return rows
    def visit_row(self, _, (consts, _1, _2)): return consts
    def visit_const(self, node, _): return float(node.text)
    def visit_consts(self, _, children): return flatten(children)

    
def parse_stl(stl_str, rule="phi"):
    return STLVisitor().visit(STL_GRAMMAR[rule].parse(stl_str))


def parse_matrix(mat_str):
    return np.array(MatrixVisitor().visit(MATRIX_GRAMMAR.parse(mat_str)))


def from_yaml(content):
    g = yaml.load(content)
    g['sys'] = [parse_stl(x, rule="sys") for x in g.get('sys', [])]
    g['env'] = [parse_stl(x, rule="env") for x in g.get('env', [])]
    g['init'] = [parse_stl(x, rule="pred") for x in g['init']]
    g['state_space']['A'] = parse_matrix(g['state_space']['A'])
    g['state_space']['B'] = parse_matrix(g['state_space']['B'])

    # TODO: check num vars

    n = g['params']['num_vars']
    n_sys = g['params']['num_sys_inputs']
    n_env = g['params']['num_env_inputs']
    assert g['state_space']['A'].shape == (n, n)
    assert g['state_space']['B'].shape == (n_sys + n_env, n)
    g['u'] = []
    g['w'] = []

    return g


def main():
    with open('example1.stl', 'r') as f:
        print(from_yaml(f))

if __name__ == '__main__':
    main()
