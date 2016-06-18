# -*- coding: utf-8 -*-

# TODO: Support parsing fixed inputs
# TODO: break out into seperate library
# TODO: allow parsing multiple ors & ands together

from functools import partialmethod
from collections import namedtuple
import operator as op
from math import ceil

from parsimonious import Grammar, NodeVisitor
from funcy import cat, flatten, pluck_attr
import yaml
import numpy as np

from blustl import stl
from blustl.game import Phi, SS, Dynamics, Game

STL_GRAMMAR = Grammar(u'''
phi = (g / f / lineq / or / and / paren_phi) (__ rank)?
rank = "[" const "]"

paren_phi = "(" __ phi __ ")"
pred = id _ op _ const_or_unbound

or = paren_phi _ ("∨" / "or") _ paren_phi
and = paren_phi _ ("∧" / "and") _ paren_phi

f = F interval phi
g = G interval phi

F = "F" / "◇"
G = "G" / "□"
interval = "[" __ const_or_unbound __ "," __ const_or_unbound __ "]"

const_or_unbound = unbound / const

lineq = terms _ op _ const_or_unbound
term = ((dt __ "*" __)? const __ "*" __)? id
terms = (term __ ("+"/"-") __ terms) / term
dt = "dt"

unbound = "?"
id = ("x" / "u" / "w") ~r"\d+"
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
        return children[0][0]

    def visit_paren_phi(self, _, children):
        return children[2]

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
        var_kind, *iden = name.text
        return stl.Var(stl.str_to_varkind[var_kind] ,int("".join(iden)))

    def visit_const(self, const, children):
        return float(const.text)

    def visit_dt(self, _, _1):
        return "dt"

    def visit_term(self, _, children):
        coeffs, var = children
        if coeffs:
            dt, c, *_ = coeffs[0]
        else:
            dt, c = False, 1
        # TODO: add symbolic val for dt
        
        return stl.Term(bool(dt), c, var)

    def visit_terms(self, _, children):
        if isinstance(children[0], list):
            term, *_, terms = children[0]
            return [term] + terms
        else:
            return children

    def visit_lineq(self, _, children):
        terms, _1, op, _2, const = children
        return stl.LinEq(terms, op, const[0])



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


def parse_stl(stl_str, rule="phi") -> "STL":
    return STLVisitor().visit(STL_GRAMMAR[rule].parse(stl_str))


def parse_matrix(mat_str) -> np.array:
    return np.array(MatrixVisitor().visit(MATRIX_GRAMMAR.parse(mat_str)))


def from_yaml(content) -> Game:
    g = yaml.load(content)
    sys = tuple(parse_stl(x) for x in g.get('sys', []))
    env = tuple(parse_stl(x) for x in g.get('env', []))
    init = [parse_stl(x) for x in g['init']]
    phi = Phi(sys, env, init)
    ss = SS(*map(parse_matrix, op.itemgetter('A', 'B')(g['state_space'])))
    width = g.get('explore_width', 5)
    dyn = Dynamics(ss, g['num_vars'], g['num_sys_inputs'], g['num_env_inputs'])
    dt = int(g['dt'])
    steps = int(ceil(int(g['time_horizon']) / dt))
    
    assert ss.A.shape == (dyn.n_vars, dyn.n_vars)
    assert ss.B.shape == (dyn.n_vars, dyn.n_sys + dyn.n_env)

    return Game(phi, dyn, width, dt, steps)

if __name__ == '__main__':
    main()
