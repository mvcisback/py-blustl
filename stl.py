# -*- coding: utf-8 -*-
import pyast as ast

class STL(ast.Node): pass

class Pred(STL):
    lit = ast.field((int,))
    op = ast.field(("<", ">", ">=", "<="))
    const = ast.field((int, float))

    def __repr__(self):
        return "x{} {} {}".format(self.lit, self.op, self.const)

class Or(STL):
    left = ast.field(STL)
    right = ast.field(STL)

    def __repr__(self): return "({}) ∨ ({})".format(self.left, self.right)

class Interval(ast.Node):
    lower = ast.field((int, float))
    upper = ast.field((int, float))

    def __repr__(self): return "[{},{}]".format(self.lower, self.upper)

class F(STL):
    arg = ast.field(STL)
    interval = ast.field(Interval)
    
    def __repr__(self): return "⋄{}({})".format(self.interval, self.arg)

# -*- coding: utf-8 -*-
import pyast as ast

class STL(ast.Node): pass

class Pred(STL):
    lit = ast.field((int,))
    op = ast.field(("<", ">", ">=", "<="))
    const = ast.field((int, float))

    def __repr__(self):
        return "x{} {} {}".format(self.lit, self.op, self.const)

class Or(STL):
    left = ast.field(STL)
    right = ast.field(STL)

    def __repr__(self): 
        return "({}) ∨ ({})".format(self.left, self.right)

class And(STL):
    left = ast.field(STL)
    right = ast.field(STL)

    def __repr__(self): 
        return "({}) ∧ ({})".format(self.left, self.right)

class Interval(ast.Node):
    lower = ast.field((int, float))
    upper = ast.field((int, float))

    def __repr__(self): return "[{},{}]".format(self.lower, self.upper)

class F(STL):
    arg = ast.field(STL)
    interval = ast.field(Interval)
    
    def __repr__(self): return "⋄{}({})".format(self.interval, self.arg)

class G(STL):
    arg = ast.field(STL)
    interval = ast.field(Interval)
    
    def __repr__(self): return "□{}({})".format(self.interval, self.arg)
    
class Neg(STL):
    arg = ast.field(STL)
    
    def __repr__(self): return "¬({})".format(self.arg)
