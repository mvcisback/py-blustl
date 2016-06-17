# -*- coding: utf-8 -*-
"""
"""

from collections import namedtuple
from typing import Union
from enum import Enum

VarKind = Enum("VarKind", ["x", "u", "w"])
str_to_varkind = {"x": VarKind.x, "u": VarKind.u, "w": VarKind.w}

class Pred(namedtuple('P', ['lit', 'op', 'const', 'kind'])):
    def __repr__(self):
        return "{k}{} {} {}".format(
            self.lit, self.op, self.const, k=self.kind.name)

    def children(self):
        return []


class Interval(namedtuple('I', ['lower', 'upper'])):
    def __repr__(self):
        return "[{},{}]".format(self.lower, self.upper)

    def children(self):
        return [self.lower, self.upper]


class NaryOpSTL(namedtuple('NaryOp', ['args'])):
    OP = "?"
    def __repr__(self):
        n = len(self.args)
        if n == 1:
            return "{}".format(self.args[0])
        elif self.args:
            rep = "({})" + " {op} ({})"*(len(self.args) - 1)
            return rep.format(*self.args, op=self.OP)
        else:
            return ""

    def children(self):
        return self.args


class Or(NaryOpSTL):
    OP = "∨"

class And(NaryOpSTL):
    OP = "∧"


class ModalOp(namedtuple('ModalOp', ['interval', 'arg'])):
    def children(self):
        return [self.arg]


class F(ModalOp):
    def __repr__(self):
        return "◇{}({})".format(self.interval, self.arg)


class G(ModalOp):
    def __repr__(self):
        return "□{}({})".format(self.interval, self.arg)


class Neg(namedtuple('Neg', ['arg'])):
    def __repr__(self):
        return "¬({})".format(self.arg)

    def children(self):
        return [self.arg]


def walk(stl):
    children = [stl]
    while len(children) != 0:
        node = children.pop()
        yield node
        children.extend(node.children())

def tree(stl):
    return {x:set(x.children()) for x in walk(stl) if x.children()}

STL = Union[Pred, NaryOpSTL, ModalOp, Neg]
