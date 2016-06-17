# -*- coding: utf-8 -*-
from blustl.stl_parser import parse_stl, from_yaml
from blustl import stl
from nose2.tools import params
import unittest
from glob import glob

def main():
    with open('examples/example1.stl', 'r') as f:
        print(from_yaml(f))

ex1 = ('x1 > 2', stl.Pred(1, ">", 2., stl.VarKind.x))
i1 = stl.Interval(0., 1.)
i2 = stl.Interval(2., 3.)
ex2 = ('◇[0,1](x1 > 2)', stl.F(i1, ex1[1]))
ex3 = ('□[2,3]◇[0,1](x1 > 2)', stl.G(i2, ex2[1]))
ex4 = ('(x1 > 2) or ((x1 > 2) or (x1 > 2))', 
       stl.Or((ex1[1], ex1[1], ex1[1])))
 
example_ymls = glob('examples/*')

class TestSTLParser(unittest.TestCase):
    @params(ex1, ex2, ex3, ex4)
    def test_stl(self, phi_str, phi):
        self.assertEqual(parse_stl(phi_str), phi)
    
    @params(*example_ymls)
    def test_from_yaml_smoketest(self, yml_path):
        with open(yml_path) as f:
            from_yaml(f)

