from functools import singledispatch
import operator as op

import numpy as np
from lenses import lens

import stl
import pyeda

@singledispatch
def encode_stl(stl):
	raise NotImplementedError
	
def bool_op(stl, conjunction = False):
	f = And if conjunction else Or
	val = 1 if conjunction else 0
	def encode_boolop(x,t):
		for tau in t:
			func.append(expr(val))
			for arg in stl.args:
				func[tau] = f(func[tau], encode_stl(arg)(x, tau), simplify = True)
		return func
	return encode_boolop
	
@encode_stl.register(stl.Or)
def _(stl):
	return bool_op(stl, conjunction = False)
	
@encode_stl.register(stl.And)
def _(stl):
	return bool_op(stl, conjunction = True)

def temp_op(stl, lo, hi, conjunction = False):
	f = And if conjunction else Or
	val = 1 if conjunction else 0
	def encode_tempop(x,t):
		for tau in t:
			tau_t = [min(tau + t2, x.index[-1]) for t2 in x[lo:hi].index]
            func.append(expr(val))
			for tau_hat in tau_t:
				func[tau] = f(func[tau], encode_stl(stl.arg)(x, tau_hat), simplify = True)
		return func
	return encode_tempop
    
@encode_stl.register(stl.F)
def _(stl):
    lo, hi = stl.interval
    return temp_op(stl, lo, hi, conjunction=False)
	
@encode_stl.register(stl.G)
def _(stl):
    lo, hi = stl.interval
    return temp_op(stl, lo, hi, conjunction=True)
    
@encode_stl.register(stl.Neg)
def _(stl):
    def neg_op(x, t):
        func = []
        for tau in t:
            func.append(Not(encode_stl(stl.arg)(x,t)))
        return func
    return neg_op
    
@encode_stl.register(stl.AtomicPred)
def _(stl):
    def encode_ap(x, t):
        func = []
        for tau in t: 
            func.append(exprvar('x',(stl.id, tau)))
        return func
    return encode_ap
    
@encode_stl.register(stl.LinEq)
def _(stl):
    op = op_lookup[stl.op]
    def encode_le(x,t):
        func = []
        for tau in t:
            func.append(exprvar('x'+str(op)+str(stl.const), (stl.id, tau)))
        return func
    return encode_le
	