# -*- coding: utf-8 -*-
import pyast as ast

class STL(ast.Node): pass

class Path_STL(STL):
    def children(self): return [self.arg]

class BinOpSTL(STL):
    def children(self): return [self.left, self.right]

class Pred(STL):
    lit = ast.field((int,))
    op = ast.field(("<", ">", ">=", "<=", "="))
    const = ast.field((int, float))

    def __repr__(self):
        return "x{} {} {}".format(self.lit, self.op, self.const)

    def children(self): return []

class Or(BinOpSTL):
    left = ast.field(STL)
    right = ast.field(STL)

    def __repr__(self): 
        return "({}) ∨ ({})".format(self.left, self.right)

class And(BinOpSTL):
    left = ast.field(STL)
    right = ast.field(STL)

    def __repr__(self): 
        return "({}) ∧ ({})".format(self.left, self.right)

class Interval(ast.Node):
    lower = ast.field((int, float))
    upper = ast.field((int, float))

    def __repr__(self): return "[{},{}]".format(self.lower, self.upper)
    
    def children(self): return [self.lower, self.upper]

class F(Path_STL):
    arg = ast.field(STL)
    interval = ast.field(Interval)
    
    def __repr__(self): return "⋄{}({})".format(self.interval, self.arg)

class G(Path_STL):
    arg = ast.field(STL)
    interval = ast.field(Interval)
    
    def __repr__(self): return "□{}({})".format(self.interval, self.arg)

    
class Neg(STL):
    arg = ast.field(STL)
    
    def __repr__(self): return "¬({})".format(self.arg)
    def children(self): return [self.arg]


def walk(stl):
    children = [stl]
    while len(children) != 0:
        node = children.pop()
        yield node
        children.extend(node.children())
