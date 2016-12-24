from functools import singledispatch
import operator as op

import numpy as np
from lenses import lens

import stl
import pyeda

def simplify_encode(simp_stl):
	return espresso_exprs(simp_stl)[0]
	
def decode(simp_stl):
    simp_stl = simplify_encode(simp_stl)
	stl_str = str(simp_stl)
	str.replace('Or', 'or')
	str.replace('And', 'and')
	str.replace('Not', 'neg')
	return stl.parse(stl_str)