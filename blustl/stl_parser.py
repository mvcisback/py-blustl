# -*- coding: utf-8 -*-
from functools import partialmethod

from parsimonious import Grammar, NodeVisitor
from funcy import cat, flatten
import yaml
import numpy as np

from blustl import stl 

# TODO: Support parsing fixed inputs

# TODO: allow parsing multiple ors & ands together
STL_GRAMMAR = Grammar(u'''
env = phi
sys = phi (_ rank)?

rank = "[" const "]"

phi = g / f / gf / fg / pred / or / and
phi2 = pred / or2 / and2
pred = id _ op _ const_or_unbound

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
interval = "[" __ const_or_unbound __ "," __ const_or_unbound __ "]"

const_or_unbound = unbound / const

unbound = "?"
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
''')


class STLVisitor(NodeVisitor):
    def generic_visit(self, _, children):
        return children

    def visit_env(self, _, phi):
        return phi

    def visit_sys(self, _, children):
        phi, maybe_rank = children
        maybe_rank = tuple(flatten(maybe_rank))
        rank = maybe_rank[0] if len(maybe_rank) > 0 else 0
        return phi

    def visit_rank(self, _, children):
        return children[1]

    def visit_phi(self, _, children):
        return children[0]

    def visit_phi2(self, _, children):
        return children[0]

    def visit_paren_phi(self, _, children):
        return children[2]

    def visit_paren_phi2(self, _, children):
        return children[2]

    def visit_pred(self, _, children):
        id, _, op, _, const = children
        return stl.Pred(id, op, const[0])

    def visit_interval(self, _, children):
        _, _, left, _, _, _, right, _, _ = children
        return stl.Interval(left[0], right[0])

    def visit_unbound(self, node, _):
        return node.text

    def visit_f(self, _, children):
        _, interval, phi = children
        return stl.F(interval, phi)

    def visit_g(self, _, children):
        _, interval, phi = children
        return stl.G(interval, phi)

    def visit_fg(self, _, children):
        _, i1, _, i2, p = children
        return stl.F(i1, stl.G(i2, p))

    def visit_gf(self, _, children):
        _1, i1, _2, i2, p = children
        return stl.G(i1, stl.F(i2, p))

    def binop_visiter(self, _, children, op):
        phi1, _, _, _, phi2 = children
        argL = list(phi1.args) if isinstance(phi1, op) else [phi1]
        argR = list(phi2.args) if isinstance(phi2, op) else [phi2]
        return op(tuple(argL + argR))

    visit_or = partialmethod(binop_visiter, op=stl.Or)
    visit_or2 = visit_or
    visit_and = partialmethod(binop_visiter, op=stl.And)
    visit_and2 = visit_and

    def visit_op(self, op, _):
        return op.text

    def visit_id(self, name, children):
        return int(name.text[1:])

    def visit_const(self, const, children):
        return float(const.text)


class MatrixVisitor(NodeVisitor):
    def generic_visit(self, _, children):
        return children

    def visit_matrix(self, _, children):
        _, _, rows, _, _ = children
        return rows

    def visit_row(self, _, children):
        consts, _, _ = children
        return consts

    def visit_const(self, node, _):
        return float(node.text)

    def visit_consts(self, _, children):
        return list(flatten(children))


def parse_stl(stl_str, rule="phi"):
    return STLVisitor().visit(STL_GRAMMAR[rule].parse(stl_str))


def parse_matrix(mat_str):
    return np.array(MatrixVisitor().visit(MATRIX_GRAMMAR.parse(mat_str)))


def from_yaml(content):
    g = yaml.load(content)
    g['sys'] = tuple(parse_stl(x, rule="sys") for x in g.get('sys', []))
    g['env'] = tuple(parse_stl(x, rule="env") for x in g.get('env', []))
    g['init'] = [parse_stl(x, rule="pred") for x in g['init']]
    g['state_space']['A'] = parse_matrix(g['state_space']['A'])
    g['state_space']['B'] = parse_matrix(g['state_space']['B'])
    # TODO: is there a more principled way to do this?
    g['explore_width'] = g.get('explore_width', 5)

    # TODO: check num vars
    n = g['num_vars']
    n_sys = g['num_sys_inputs']
    n_env = g['num_env_inputs']
    assert g['state_space']['A'].shape == (n, n)
    assert g['state_space']['B'].shape == (n, n_sys + n_env)

    return g

if __name__ == '__main__':
    main()
